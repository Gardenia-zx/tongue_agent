from app.agent.state import AgentState


async def tongue_analysis_node(state: AgentState) -> AgentState:
    return {
        **state,
        "current_node": "tongue_analysis_node",
        "response_message": {
            "role": "assistant",
            "content_type": "text",
            "content": "可以开始舌象分析。请上传舌象图片，或继续当前未完成的分析任务。",
        },
        "next_action": {
            "type": "START_OR_RESUME_TONGUE_ANALYSIS",
            "payload": {
                "report_id": state.get("report_id"),
                "task_id": state.get("task_id"),
                "thread_id": state.get("thread_id"),
            },
        },
    }
