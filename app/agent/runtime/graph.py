import asyncio
import hashlib
import json
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.agent.context_builder import with_prompt_context
from app.agent.nodes.agent_loop_node import (
    AGENT_LOOP_SYSTEM_PROMPT,
    _assistant_tool_call_message,
    _build_initial_messages,
    _execute_tool_call,
    _fallback_from_last_tool,
    _merge_agent_loop_into_next_action,
    _run_rule_based_fallback,
    _state_from_final_answer,
    _summarize_state,
    _tool_schemas,
)
from app.agent.state import AgentState
from app.core.config import get_settings
from app.integrations.model_gateway import ModelGatewayError, get_chat_model_client

RuntimeRoute = Literal["VALIDATE_PLAN", "FINALIZE", "FALLBACK"]
ValidationRoute = Literal["EXECUTE_TOOL", "CHECK_BUDGET", "FALLBACK"]
ExecutorRoute = Literal["EXECUTE_TOOL", "CHECK_BUDGET", "FALLBACK"]
BudgetRoute = Literal["CONTINUE", "FALLBACK"]

DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_TOOL_CALLS = 10
DEFAULT_MAX_TOOL_CALLS_PER_PLAN = 4
DEFAULT_TOOL_TIMEOUT_SECONDS = 45.0
MAX_TOOL_ARGUMENT_BYTES = 16_384
MAX_OBSERVATION_STRING_LENGTH = 2_000


def _runtime(state: AgentState) -> dict[str, Any]:
    value = state.get("agent_loop") or {}
    return dict(value) if isinstance(value, dict) else {}


def _known_tool_names() -> set[str]:
    return {
        str(schema.get("function", {}).get("name") or "")
        for schema in _tool_schemas()
        if schema.get("function", {}).get("name")
    }


def _tool_call_id(raw_tool_call: dict[str, Any], *, iteration: int, index: int) -> str:
    value = raw_tool_call.get("id")
    if value:
        return str(value)
    return f"runtime-{iteration}-{index}"


def _parse_arguments_strict(raw_tool_call: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    function = raw_tool_call.get("function") or {}
    raw_arguments = function.get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        if len(raw_arguments.encode("utf-8")) > MAX_TOOL_ARGUMENT_BYTES:
            return {}, "tool_arguments_too_large"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}, "invalid_tool_arguments_json"
    else:
        return {}, "invalid_tool_arguments_type"

    if not isinstance(arguments, dict):
        return {}, "tool_arguments_must_be_object"
    return arguments, None


def _call_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<omitted>"
    if isinstance(value, str):
        if len(value) <= MAX_OBSERVATION_STRING_LENGTH:
            return value
        return f"{value[:MAX_OBSERVATION_STRING_LENGTH]}..."
    if isinstance(value, list):
        return [_compact_value(item, depth=depth + 1) for item in value[:8]]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 24:
                compact["_truncated"] = True
                break
            compact[str(key)] = _compact_value(item, depth=depth + 1)
        return compact
    return value


def _append_tool_message(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: str,
    tool_name: str,
    result: dict[str, Any],
) -> None:
    observation = _compact_value(result)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(observation, ensure_ascii=False, default=str),
        }
    )


def _append_rejected_call(
    runtime: dict[str, Any],
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    error: str,
) -> None:
    result = {
        "status": "REJECTED",
        "tool_name": tool_name,
        "error": error,
        "retryable": True,
    }
    messages = list(runtime.get("messages") or [])
    _append_tool_message(
        messages,
        tool_call_id=tool_call_id,
        tool_name=tool_name or "unknown_tool",
        result=result,
    )
    records = list(runtime.get("tool_calls") or [])
    records.append(
        {
            "iteration": runtime.get("iteration", 0),
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "output": result,
            "status": "REJECTED",
        }
    )
    runtime["messages"] = messages
    runtime["tool_calls"] = records


def _finish_runtime(
    state: AgentState,
    *,
    execution_status: str,
    finish_reason: str,
) -> AgentState:
    runtime = _runtime(state)
    tool_calls = list(runtime.get("tool_calls") or [])
    runtime.update(
        {
            "schema_version": "2.0",
            "mode": "langgraph_runtime_subgraph",
            "status": "COMPLETED",
            "execution_status": execution_status,
            "finish_reason": finish_reason,
            "selected_tool": tool_calls[-1].get("tool_name") if tool_calls else None,
            "pending_tool_calls": [],
            "planner_route": None,
            "validation_route": None,
            "executor_route": None,
            "budget_route": None,
        }
    )
    # The full prompt/tool transcript is checkpointed during execution. Remove it
    # from the terminal state to keep the response snapshot and later checkpoints compact.
    runtime.pop("messages", None)
    runtime.pop("raw_tool_calls", None)
    runtime.pop("final_raw_content", None)

    next_action = _merge_agent_loop_into_next_action(state.get("next_action"), runtime)
    return {
        **state,
        "agent_loop": runtime,
        "next_action": next_action,
    }


async def loop_init_node(state: AgentState) -> AgentState:
    prepared = with_prompt_context(
        state,
        mode="FULL_FOR_NODE",
        system_prompt=AGENT_LOOP_SYSTEM_PROMPT,
        node_name="agent_runtime.loop_init",
        include_long_term_memory=True,
    )
    runtime = {
        "schema_version": "2.0",
        "mode": "langgraph_runtime_subgraph",
        "status": "RUNNING",
        "execution_status": "RUNNING",
        "turn_id": prepared.get("turn_id"),
        "iteration": 0,
        "model_calls": 0,
        "tool_calls": [],
        "pending_tool_calls": [],
        "messages": _build_initial_messages(prepared),
        "call_fingerprint_counts": {},
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
        "finish_reason": None,
    }
    return {
        **prepared,
        "current_node": "agent_runtime.loop_init",
        "agent_loop": runtime,
    }


async def planner_node(state: AgentState) -> AgentState:
    runtime = _runtime(state)
    iteration = int(runtime.get("iteration") or 0)
    max_iterations = int(runtime.get("max_iterations") or DEFAULT_MAX_ITERATIONS)
    if iteration >= max_iterations:
        runtime["planner_route"] = "FALLBACK"
        runtime["finish_reason"] = "max_iterations_reached"
        return {
            **state,
            "current_node": "agent_runtime.planner",
            "agent_loop": runtime,
        }

    settings = get_settings()
    messages = list(runtime.get("messages") or [])
    runtime["iteration"] = iteration + 1
    runtime["model_calls"] = int(runtime.get("model_calls") or 0) + 1

    try:
        assistant_message = await get_chat_model_client().chat(
            messages=messages,
            tools=_tool_schemas(),
            tool_choice="auto",
            temperature=settings.chat_model_temperature,
            max_tokens=settings.chat_model_max_tokens,
        )
    except ModelGatewayError as exc:
        runtime["planner_route"] = "FALLBACK"
        runtime["finish_reason"] = "model_error_rule_fallback"
        runtime["last_model_error"] = type(exc).__name__
        return {
            **state,
            "current_node": "agent_runtime.planner",
            "agent_loop": runtime,
        }

    model_tool_calls = assistant_message.get("tool_calls") or []
    if model_tool_calls:
        messages.append(_assistant_tool_call_message(assistant_message))
        runtime["messages"] = messages
        runtime["raw_tool_calls"] = list(model_tool_calls)
        runtime["planner_route"] = "VALIDATE_PLAN"
        return {
            **state,
            "current_node": "agent_runtime.planner",
            "agent_loop": runtime,
        }

    content = assistant_message.get("content")
    runtime["final_raw_content"] = content if isinstance(content, str) else ""
    runtime["planner_route"] = "FINALIZE"
    runtime["finish_reason"] = "final_answer"
    return {
        **state,
        "current_node": "agent_runtime.planner",
        "agent_loop": runtime,
    }


def route_after_planner(state: AgentState) -> RuntimeRoute:
    route = str(_runtime(state).get("planner_route") or "FALLBACK")
    if route in {"VALIDATE_PLAN", "FINALIZE", "FALLBACK"}:
        return route  # type: ignore[return-value]
    return "FALLBACK"


async def plan_validator_node(state: AgentState) -> AgentState:
    runtime = _runtime(state)
    raw_tool_calls = list(runtime.get("raw_tool_calls") or [])
    known_tools = _known_tool_names()
    pending: list[dict[str, Any]] = []
    iteration = int(runtime.get("iteration") or 0)
    fingerprint_counts = dict(runtime.get("call_fingerprint_counts") or {})

    for index, raw_tool_call in enumerate(raw_tool_calls):
        tool_call_id = _tool_call_id(raw_tool_call, iteration=iteration, index=index)
        function = raw_tool_call.get("function") or {}
        tool_name = str(function.get("name") or "")
        arguments, parse_error = _parse_arguments_strict(raw_tool_call)

        error = parse_error
        if index >= DEFAULT_MAX_TOOL_CALLS_PER_PLAN:
            error = "too_many_tool_calls_in_single_plan"
        elif not tool_name or tool_name not in known_tools:
            error = "unknown_tool"

        fingerprint = _call_fingerprint(tool_name, arguments)
        if not error and int(fingerprint_counts.get(fingerprint) or 0) >= 2:
            error = "duplicate_tool_call_limit_reached"

        # A report must never be generated before tongue features are available.
        if (
            not error
            and tool_name == "tongue_report_generate_tool"
            and not state.get("tongue_features")
        ):
            error = "tongue_features_required"

        if error:
            _append_rejected_call(
                runtime,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                error=error,
            )
            continue

        pending.append(
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "fingerprint": fingerprint,
            }
        )

    runtime["raw_tool_calls"] = []
    runtime["pending_tool_calls"] = pending
    runtime["validation_route"] = "EXECUTE_TOOL" if pending else "CHECK_BUDGET"
    return {
        **state,
        "current_node": "agent_runtime.plan_validator",
        "agent_loop": runtime,
    }


def route_after_validation(state: AgentState) -> ValidationRoute:
    route = str(_runtime(state).get("validation_route") or "FALLBACK")
    if route in {"EXECUTE_TOOL", "CHECK_BUDGET", "FALLBACK"}:
        return route  # type: ignore[return-value]
    return "FALLBACK"


async def tool_executor_node(state: AgentState) -> AgentState:
    runtime = _runtime(state)
    pending = list(runtime.get("pending_tool_calls") or [])
    if not pending:
        runtime["executor_route"] = "CHECK_BUDGET"
        return {
            **state,
            "current_node": "agent_runtime.tool_executor",
            "agent_loop": runtime,
        }

    tool_call = pending.pop(0)
    tool_call_id = str(tool_call.get("tool_call_id") or "")
    tool_name = str(tool_call.get("tool_name") or "")
    arguments = tool_call.get("arguments") or {}
    fingerprint = str(tool_call.get("fingerprint") or "")

    try:
        next_state, tool_result = await asyncio.wait_for(
            _execute_tool_call(
                state=state,
                tool_name=tool_name,
                arguments=arguments,
            ),
            timeout=DEFAULT_TOOL_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        next_state = state
        tool_result = {
            "status": "FAILED",
            "tool_name": tool_name,
            "error": "tool_timeout",
            "retryable": True,
        }
    except Exception as exc:  # noqa: BLE001 - tool failures become observations
        next_state = state
        tool_result = {
            "status": "FAILED",
            "tool_name": tool_name,
            "error": type(exc).__name__,
            "retryable": True,
        }

    messages = list(runtime.get("messages") or [])
    _append_tool_message(
        messages,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=tool_result,
    )
    records = list(runtime.get("tool_calls") or [])
    records.append(
        {
            "iteration": runtime.get("iteration", 0),
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "output": _compact_value(tool_result),
            "status": tool_result.get("status"),
        }
    )
    fingerprint_counts = dict(runtime.get("call_fingerprint_counts") or {})
    if fingerprint:
        fingerprint_counts[fingerprint] = int(fingerprint_counts.get(fingerprint) or 0) + 1

    runtime.update(
        {
            "messages": messages,
            "tool_calls": records,
            "pending_tool_calls": pending,
            "call_fingerprint_counts": fingerprint_counts,
            "last_observation": _compact_value(tool_result),
            "executor_route": "EXECUTE_TOOL" if pending else "CHECK_BUDGET",
        }
    )
    return {
        **next_state,
        "current_node": "agent_runtime.tool_executor",
        "agent_loop": runtime,
    }


def route_after_executor(state: AgentState) -> ExecutorRoute:
    route = str(_runtime(state).get("executor_route") or "FALLBACK")
    if route in {"EXECUTE_TOOL", "CHECK_BUDGET", "FALLBACK"}:
        return route  # type: ignore[return-value]
    return "FALLBACK"


async def budget_guard_node(state: AgentState) -> AgentState:
    runtime = _runtime(state)
    iteration = int(runtime.get("iteration") or 0)
    max_iterations = int(runtime.get("max_iterations") or DEFAULT_MAX_ITERATIONS)
    tool_call_count = len(runtime.get("tool_calls") or [])
    max_tool_calls = int(runtime.get("max_tool_calls") or DEFAULT_MAX_TOOL_CALLS)

    if tool_call_count >= max_tool_calls:
        runtime["budget_route"] = "FALLBACK"
        runtime["finish_reason"] = "max_tool_calls_reached"
    elif iteration >= max_iterations:
        runtime["budget_route"] = "FALLBACK"
        runtime["finish_reason"] = "max_iterations_reached"
    else:
        runtime["budget_route"] = "CONTINUE"

    return {
        **state,
        "current_node": "agent_runtime.budget_guard",
        "agent_loop": runtime,
    }


def route_after_budget(state: AgentState) -> BudgetRoute:
    route = str(_runtime(state).get("budget_route") or "FALLBACK")
    return "CONTINUE" if route == "CONTINUE" else "FALLBACK"


async def answer_finalize_node(state: AgentState) -> AgentState:
    runtime = _runtime(state)
    content = str(runtime.get("final_raw_content") or "")
    final_state = _state_from_final_answer(state, content)
    final_state = {
        **final_state,
        "current_node": "agent_runtime.answer_finalize",
    }
    return _finish_runtime(
        final_state,
        execution_status="SUCCEEDED",
        finish_reason=str(runtime.get("finish_reason") or "final_answer"),
    )


async def fallback_node(state: AgentState) -> AgentState:
    runtime = _runtime(state)
    finish_reason = str(runtime.get("finish_reason") or "runtime_fallback")
    try:
        fallback_state = await asyncio.wait_for(
            _run_rule_based_fallback(state),
            timeout=DEFAULT_TOOL_TIMEOUT_SECONDS,
        )
        fallback_output = _summarize_state(fallback_state)
    except Exception as exc:  # noqa: BLE001 - final deterministic degradation path
        fallback_state = _fallback_from_last_tool(state)
        fallback_output = {
            "status": "FAILED",
            "error": type(exc).__name__,
        }

    records = list(runtime.get("tool_calls") or [])
    records.append(
        {
            "iteration": runtime.get("iteration", 0),
            "tool_call_id": "rule-based-fallback",
            "tool_name": "rule_based_fallback",
            "arguments": {},
            "output": _compact_value(fallback_output),
            "status": "COMPLETED",
        }
    )
    runtime["tool_calls"] = records
    fallback_state = {
        **fallback_state,
        "current_node": "agent_runtime.fallback",
        "agent_loop": runtime,
    }
    return _finish_runtime(
        fallback_state,
        execution_status="DEGRADED",
        finish_reason=finish_reason,
    )


def build_agent_runtime_subgraph():
    """Build the checkpointable model/tool loop used as a node in the outer graph.

    The subgraph is intentionally compiled without its own checkpointer. LangGraph
    propagates the parent graph's persistence configuration into the child graph,
    so every planner, validation, tool execution and budget step can be resumed.
    """

    graph = StateGraph(AgentState)
    graph.add_node("loop_init", loop_init_node)
    graph.add_node("planner", planner_node)
    graph.add_node("plan_validator", plan_validator_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("budget_guard", budget_guard_node)
    graph.add_node("answer_finalize", answer_finalize_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("loop_init")
    graph.add_edge("loop_init", "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "VALIDATE_PLAN": "plan_validator",
            "FINALIZE": "answer_finalize",
            "FALLBACK": "fallback",
        },
    )
    graph.add_conditional_edges(
        "plan_validator",
        route_after_validation,
        {
            "EXECUTE_TOOL": "tool_executor",
            "CHECK_BUDGET": "budget_guard",
            "FALLBACK": "fallback",
        },
    )
    graph.add_conditional_edges(
        "tool_executor",
        route_after_executor,
        {
            "EXECUTE_TOOL": "tool_executor",
            "CHECK_BUDGET": "budget_guard",
            "FALLBACK": "fallback",
        },
    )
    graph.add_conditional_edges(
        "budget_guard",
        route_after_budget,
        {
            "CONTINUE": "planner",
            "FALLBACK": "fallback",
        },
    )
    graph.add_edge("answer_finalize", END)
    graph.add_edge("fallback", END)
    return graph
