"""Production-oriented Agent runtime subgraph.

The runtime package owns the model/tool observation loop. The outer Agent graph
continues to own request routing, memory lifecycle, safety branches and final
turn persistence.
"""

from typing import Any

from app.agent.nodes import agent_loop_node as legacy_agent_loop


def _merge_next_action_compat(
    next_action: Any,
    agent_loop: dict[str, Any],
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Merge runtime metadata without depending on a legacy private signature.

    Some development branches added a keyword-only ``state`` argument to the
    legacy helper while the runtime subgraph was created against the previous
    two-argument version. Keeping this compatibility function at the package
    boundary makes both layouts behave identically and prevents finalization
    from failing after the model has already produced an answer.
    """

    del state  # Reserved for compatibility with the newer legacy signature.
    if not isinstance(next_action, dict):
        return {
            "type": "RESPOND_TO_USER",
            "payload": {
                "status": "COMPLETED",
                "agent_loop": agent_loop,
            },
        }

    payload = next_action.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    return {
        **next_action,
        "payload": {
            **payload,
            "agent_loop": agent_loop,
        },
    }


# The runtime graph currently imports several legacy helpers. Patch only this
# compatibility seam before importing the graph so both helper signatures are
# supported. A later cleanup can move all shared helpers into public modules.
legacy_agent_loop._merge_agent_loop_into_next_action = _merge_next_action_compat

from app.agent.runtime.graph import build_agent_runtime_subgraph  # noqa: E402

__all__ = ["build_agent_runtime_subgraph"]
