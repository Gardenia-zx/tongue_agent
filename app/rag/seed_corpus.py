from app.schemas.rag import RagSourceDocument


SEED_METADATA = {
    "source_repo": "internal_seed",
    "source_path": "app/rag/seed_corpus.py",
    "source_commit": None,
    "license": "internal_mvp_seed",
    "content_storage": "inline",
    "content_path": "app/rag/seed_corpus.py",
}


SEED_DOCUMENTS = [
    RagSourceDocument(
        doc_id="seed_tongue_overview",
        title="舌象观察的一般原则",
        content=(
            "舌象观察通常会关注舌质、舌色、舌形、舌苔颜色、舌苔厚薄、润燥和分布等信息。"
            "这些信息可以作为健康管理和中医体质讨论的参考，但不能单独用于疾病诊断。"
            "日常拍摄舌象时，应尽量选择自然光，避免滤镜、强光反射和刚吃过有色食物后的影响。"
        ),
        tags=["舌象", "健康管理", "基础知识"],
        metadata=SEED_METADATA,
    ),
    RagSourceDocument(
        doc_id="seed_dampness",
        title="湿气重与痰湿相关表达",
        content=(
            "“湿气重”是日常语言中常见的中医健康表达，常被用来描述身体困重、乏力、食欲差、"
            "大便黏滞、舌苔偏厚或偏腻等感受。它不是现代医学诊断名称，也不能仅凭一个症状判断。"
            "如果只是想做健康管理，可以从规律作息、清淡饮食、适度运动、减少过甜过油食物等方向观察调整。"
            "如果不适明显、持续加重，建议线下就医。"
        ),
        tags=["湿气重", "痰湿", "健康知识"],
        metadata=SEED_METADATA,
    ),
    RagSourceDocument(
        doc_id="seed_teeth_marks",
        title="齿痕舌的一般说明",
        content=(
            "齿痕舌通常指舌体边缘可以看到类似牙齿压痕的痕迹。它可能与舌体偏胖、口腔空间、"
            "咬合习惯、短期水肿感或个体差异有关。中医讨论中常会把齿痕舌与脾虚、湿困等概念联系起来，"
            "但需要结合整体表现和专业判断，不能单独作为诊断依据。"
        ),
        tags=["齿痕舌", "舌形", "健康知识"],
        metadata=SEED_METADATA,
    ),
    RagSourceDocument(
        doc_id="seed_greasy_coating",
        title="腻苔和厚苔的区别",
        content=(
            "厚苔主要描述舌苔覆盖较厚，舌面底色不容易看清；腻苔主要描述舌苔颗粒细腻、黏腻、不易刮净的感觉。"
            "二者可能同时出现。饮食、口腔清洁、睡眠、胃肠状态和近期身体状况都可能影响舌苔表现。"
            "如果舌苔变化持续存在并伴随明显不适，应结合线下检查或医生建议。"
        ),
        tags=["腻苔", "厚苔", "舌苔"],
        metadata=SEED_METADATA,
    ),
    RagSourceDocument(
        doc_id="seed_yellow_white_coating",
        title="白苔和黄苔的一般理解",
        content=(
            "白苔和黄苔是舌苔颜色的常见描述。白苔较常见，可能与正常舌苔、饮食、口腔清洁等有关；"
            "黄苔在中医语境中常被拿来讨论热象、湿热等倾向，但不能仅凭颜色判断具体问题。"
            "舌苔颜色容易受到食物、饮料、药物和拍摄光线影响，观察时应尽量排除这些干扰。"
        ),
        tags=["白苔", "黄苔", "舌苔"],
        metadata=SEED_METADATA,
    ),
    RagSourceDocument(
        doc_id="seed_lifestyle",
        title="舌象健康管理的生活方式建议",
        content=(
            "一般健康管理可以从作息、饮食、运动和情绪压力入手。建议保持规律睡眠，减少长期熬夜；"
            "饮食上避免长期过油、过甜、过辣和暴饮暴食；根据体力进行稳定、适度的运动；"
            "如果伴随持续疼痛、发热、明显消瘦、胸痛、呼吸困难等情况，应及时寻求线下医疗帮助。"
        ),
        tags=["生活方式", "健康管理", "安全提示"],
        metadata=SEED_METADATA,
    ),
]
