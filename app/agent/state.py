from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph Agent 在一次请求和一次会话中的共享状态。

    说明：
    - `total=False` 表示所有字段都是可选字段，不同节点可以只读写自己关心的字段。
    - 状态会被 LangGraph Checkpointer 持久化，因此字段命名要保持稳定。
    - 新字段优先放入 `extensions` 或现有结构化字段中，避免频繁破坏状态契约。
    """

    # 状态 Schema 版本，用于后续兼容旧状态或迁移状态结构。
    schema_version: str

    # 全链路追踪 ID，由前端或 Java 后端生成，用于串联前端、后端、Agent、模型服务日志。
    trace_id: str
    # 单次 Agent 请求 ID，用于区分同一 trace 下的多次请求。
    request_id: str
    # 当前用户 ID，长期记忆、用户画像、报告权限隔离都以它为核心。
    user_id: int
    # 租户 ID，用于跨租户隔离 Agent 状态、记忆和幂等记录。
    tenant_id: str
    # LangGraph 会话 ID，也是并发锁和短期记忆隔离的核心键。
    thread_id: str
    # 会话 Epoch。重置会话时递增，避免旧 Checkpoint 污染新会话。
    thread_epoch: int
    # 单轮对话幂等 ID，由 Java 入口生成，Python 持久化在 agent_turn_record。
    turn_id: str
    user_message_id: str
    assistant_message_id: str

    # 业务对话 ID，由前端或 Java 后端传入，用于归档同一业务会话。
    conversation_id: str
    # 舌象报告 ID，舌象分析、报告解释、历史报告查询时使用。
    report_id: int
    # 当前分析任务 ID，用于追踪一次模型识别或报告生成任务。
    task_id: int
    # 当前任务版本号，用于后续报告重跑、模型重算或报告修订。
    task_version: int

    # 当前轮用户消息，通常包含 role、content_type、content、attachments。
    message: dict[str, Any]
    # 客户端上下文，例如页面来源、设备类型、语言、图片路径、当前报告 ID 等。
    client_context: dict[str, Any]
    # Java 后端或 context_builder 汇总后的上下文包，例如历史消息、当前报告、会话摘要。
    context_bundle: dict[str, Any]
    # 给 LLM Prompt 使用的结构化上下文，通常由上下文构建节点整理。
    prompt_context: dict[str, Any]
    # 查询改写和 RAG 检索使用的上下文，例如原始问题、补全后的查询、引用对象。
    query_context: dict[str, Any]
    # 当前 Turn 的临时上下文，按 turn_id 隔离，避免读取上一轮解析结果。
    current_turn: dict[str, Any]
    # Turn 生命周期守卫，记录状态清理、响应归属校验和修复结果。
    turn_guard: dict[str, Any]
    # Runtime 上下文披露策略，决定是否允许加载报告和历史对话。
    context_policy: dict[str, Any]
    # Runtime 和领域节点共享的业务上下文。
    business_context: dict[str, Any]
    # 本次请求的运行选项，例如 memory.can_read、memory.can_write、debug 开关等。
    options: dict[str, Any]

    # 当前执行到的节点名称，主要用于调试、响应快照和链路追踪。
    current_node: str

    # 意图识别结果，包括 detected_intent、primary_intent、confidence、route_target 等。
    intent_result: dict[str, Any]
    # 安全审查结果，包括风险等级、命中规则、安全提示等。
    safety_result: dict[str, Any]

    # 图片质量检测结果，例如清晰度、曝光、是否缺少舌体等，MVP 可为空或后续启用。
    image_quality: dict[str, Any]
    # 舌象模型识别和标准化后的特征，例如 detected_feature_codes、rag_query、分组特征。
    tongue_features: dict[str, Any]

    # RAG 检索和回答上下文，包括 query、hits、answer、grounded、retrieval_engine 等。
    rag_context: dict[str, Any]
    # Agent 需要向用户追问的问题列表，用于多轮补充信息或报告生成前确认。
    questions: list[dict[str, Any]]
    # 用户对追问的回答列表，用于恢复同一 thread_id 后继续生成报告。
    answers: list[dict[str, Any]]

    # 记忆系统上下文，包括用户画像、长期记忆、摘要、短期会话缓存、写入结果等。
    memory_context: dict[str, Any]
    # Checkpointer 持久化的短期会话上下文，是当前会话历史的主事实源。
    short_term_memory: dict[str, Any]
    # Agent Loop / ReAct 模式的运行状态，例如工具调用历史、循环次数、最终回答等。
    agent_loop: dict[str, Any]

    # 草稿报告，通常由舌象分析或报告节点生成，返回 Java 后端保存。
    draft_report: dict[str, Any]
    # 最终报告，后续审核、确认或正式发布后写入；MVP 阶段可暂不使用。
    final_report: dict[str, Any]

    # 下一步动作描述，例如 RESPOND_TO_USER、TONGUE_FEATURES_READY、WAIT_USER_UPLOAD 等。
    next_action: dict[str, Any]
    # 返回给用户的消息体，通常包含 role、content_type、content。
    response_message: dict[str, Any]
    # 工具选择节点的结构化决策结果。
    tool_decision: dict[str, Any]
    # 回答质量与安全复核结果。
    quality_review: dict[str, Any]

    # 版本信息，例如模型版本、RAG 索引版本、Prompt 版本、报告 Schema 版本。
    versions: dict[str, Any]

    # 重试状态，例如失败节点、重试次数、上一次错误、是否允许重试。
    retry: dict[str, Any]
    # 错误列表，用于累计节点执行过程中的可观测错误，不一定都直接返回用户。
    errors: list[dict[str, Any]]

    # 扩展字段，用于临时实验、灰度能力或不稳定字段，避免破坏主状态结构。
    extensions: dict[str, Any]
