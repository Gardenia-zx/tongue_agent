from dataclasses import dataclass


@dataclass(frozen=True)
class ExpandedRagQuery:
    raw_query: str
    search_text: str
    embedding_text: str
    terms: list[str]


TCM_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("湿气重", "湿气", "湿重", "体内湿", "祛湿"),
        (
            "湿邪",
            "湿证",
            "痰湿",
            "水湿",
            "湿浊",
            "寒湿",
            "湿热",
            "湿性重浊",
            "脾虚生湿",
            "舌苔厚腻",
            "苔白腻",
            "苔黄腻",
            "身重困倦",
            "胸闷脘痞",
        ),
    ),
    (
        ("舌苔厚腻", "厚腻苔", "苔厚腻", "舌苔很厚", "舌苔很腻"),
        (
            "舌苔厚腻",
            "厚腻",
            "苔白腻",
            "苔黄腻",
            "痰湿",
            "湿浊",
            "食积",
            "湿热",
            "脾胃运化",
        ),
    ),
    (
        ("齿痕舌", "舌边齿痕", "舌头有齿痕", "舌体齿痕"),
        (
            "齿痕",
            "舌体胖嫩",
            "舌胖",
            "脾虚",
            "水湿",
            "痰湿内停",
            "脾肾阳虚",
        ),
    ),
    (
        ("舌苔发黄", "黄苔", "舌苔黄"),
        (
            "黄苔",
            "苔黄",
            "湿热",
            "热邪",
            "里热",
            "苔黄腻",
        ),
    ),
    (
        ("舌苔发白", "白苔", "舌苔白"),
        (
            "白苔",
            "苔白",
            "寒湿",
            "痰湿",
            "苔白腻",
        ),
    ),
)


def expand_rag_query(query: str) -> ExpandedRagQuery:
    raw_query = query.strip()
    compact_query = "".join(raw_query.split())
    terms: list[str] = []

    for triggers, expansions in TCM_QUERY_EXPANSIONS:
        if any(trigger in compact_query for trigger in triggers):
            for term in (*triggers, *expansions):
                if term not in terms:
                    terms.append(term)

    search_parts = [raw_query, *terms]
    search_text = " ".join(part for part in search_parts if part)
    embedding_text = search_text if terms else raw_query

    return ExpandedRagQuery(
        raw_query=raw_query,
        search_text=search_text,
        embedding_text=embedding_text,
        terms=terms,
    )
