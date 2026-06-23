import hashlib
import re
from dataclasses import dataclass
from html import unescape

from app.schemas.rag import RagChunk, RagSourceDocument


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；;!?])")
HEADING_PATTERN = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*$")
IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\([^)]+\)")
HTML_BREAK_PATTERN = re.compile(r"</(?:p|div|br|tr|li|h[1-6])\s*>", re.IGNORECASE)
HTML_CELL_PATTERN = re.compile(r"</t[dh]\s*>", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
LATEX_INLINE_PATTERN = re.compile(r"\$([^$]{1,200})\$")

NOISE_HEADING_KEYWORDS = (
    "图书在版",
    "融合出版",
    "资源访问",
    "资源下载",
    "编创委员会",
    "教材办公室",
    "名誉主任委员",
    "主任委员",
    "副主任委员",
    "办公室主任",
    "办公室成员",
    "秘书长",
    "学术秘书",
    "主审",
    "主编",
    "副主编",
    "编委",
    "坚持立德树人",
    "优化知识结构",
    "突出“三基五性”",
    "强化精品意识",
    "加强数字化建设",
)
CATALOG_HEADING_KEYWORDS = ("目录", "教材目录", "复习思考题")
TAIL_DROP_HEADING_KEYWORDS = ("索引", "参考文献")
CONTENT_START_PATTERN = re.compile(
    r"^(绪论|总论|上篇|中篇|下篇|理论篇|临床篇|基础篇|附篇|第[一二三四五六七八九十百千万0-9]+[章节篇])"
)
PAGE_NUMBER_SUFFIX_PATTERN = re.compile(r"[\s.·…,-]*(?:\?|\d{1,4})$")
CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class TextbookSection:
    heading_path: list[str]
    body: str
    content_type: str


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = IMAGE_PATTERN.sub("", text)
    text = _html_to_text(text)
    text = LATEX_INLINE_PATTERN.sub(r"\1", text)
    text = unescape(text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_text(text: str) -> str:
    text = HTML_CELL_PATTERN.sub(" | ", text)
    text = HTML_BREAK_PATTERN.sub("\n", text)
    text = HTML_TAG_PATTERN.sub("", text)
    return text


def _chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}_{chunk_index:04d}_{digest}"


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_PATTERN.split(text)
    return [part.strip() for part in parts if part.strip()]


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    return text[-overlap:].lstrip()


def _normalize_heading(raw_heading: str) -> str:
    heading = raw_heading.strip().strip("#").strip()
    heading = re.sub(r"\s+", " ", heading)
    heading = re.sub(r"^[第]?\d{1,4}\s*", "", heading)
    return heading.strip(" -—·.\t")


def _heading_level(markdown_level: int, heading: str) -> int:
    if re.match(r"^(上篇|中篇|下篇|理论篇|临床篇|基础篇|附篇|绪论|总论)$", heading):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+篇", heading):
        return 1
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+章", heading):
        return 2
    if re.match(r"^第[一二三四五六七八九十百千万0-9]+节", heading):
        return 3
    if re.match(r"^[一二三四五六七八九十百千万]+、", heading):
        return 4
    if re.match(r"^（[一二三四五六七八九十百千万0-9]+）", heading):
        return 5
    return min(max(markdown_level, 1), 6)


def _update_heading_stack(stack: list[str], level: int, heading: str) -> list[str]:
    if not heading:
        return stack
    next_stack = stack[: max(level - 1, 0)]
    next_stack.append(heading)
    return next_stack


def _heading_context(path: list[str], book_title: str) -> str:
    context = [book_title, *path]
    compact = [item for item in context if item]
    text = " > ".join(compact)
    return text[:180]


def _is_noise_heading(heading: str) -> bool:
    return any(keyword in heading for keyword in NOISE_HEADING_KEYWORDS)


def _is_catalog_heading(heading: str) -> bool:
    return any(keyword in heading for keyword in CATALOG_HEADING_KEYWORDS)


def _is_tail_drop_heading(heading: str) -> bool:
    return any(keyword in heading for keyword in TAIL_DROP_HEADING_KEYWORDS)


def _is_catalog_like_heading(heading: str) -> bool:
    return _is_catalog_heading(heading) or bool(
        PAGE_NUMBER_SUFFIX_PATTERN.search(heading) and CONTENT_START_PATTERN.match(heading)
    )


def _looks_like_toc_entry(heading: str, body: str) -> bool:
    if body.strip() and len(body.strip()) > 40:
        return _looks_like_catalog_body(body)
    if _is_catalog_heading(heading):
        return True
    if PAGE_NUMBER_SUFFIX_PATTERN.search(heading) and CONTENT_START_PATTERN.match(heading):
        return True
    if re.match(r"^\d+\s*目录$", heading):
        return True
    return False


def _looks_like_catalog_body(body: str) -> bool:
    compact_body = re.sub(r"\s+", "", body)
    compact_hits = re.findall(
        r"(?:第[一二三四五六七八九十百千万0-9]+节|[一二三四五六七八九十百千万]+、)[^。！？；;!?]{1,60}[\d?]{1,4}",
        compact_body,
    )
    if len(compact_hits) >= 2:
        return True

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    catalog_lines = 0
    for line in lines:
        compact = re.sub(r"\s+", "", line)
        if re.search(r"[\d?]{1,4}$", compact) and (
            compact.startswith(("第", "附", "上篇", "中篇", "下篇"))
            or re.match(r"^[一二三四五六七八九十百千万]+、", compact)
            or "第一节" in compact
            or "第二节" in compact
            or "第三节" in compact
        ):
            catalog_lines += 1

    return catalog_lines / len(lines) >= 0.5


def _looks_like_page_header(heading: str, body: str, book_title: str) -> bool:
    if body.strip() and len(body.strip()) > 40:
        return False
    if book_title and re.match(rf"^\d+\s*{re.escape(book_title)}$", heading):
        return True
    return False


def _is_low_value_body(text: str, *, min_chars: int = 30) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < min_chars:
        return True
    chinese_chars = len(CHINESE_CHAR_PATTERN.findall(compact))
    if chinese_chars / max(len(compact), 1) < 0.18:
        return True
    return False


def _strip_noisy_lines(text: str) -> str:
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if stripped in {"!", "#", "##", "###"}:
            continue
        if IMAGE_PATTERN.fullmatch(stripped):
            continue
        if re.fullmatch(r"[-_=*]{3,}", stripped):
            continue
        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _iter_sections(text: str, *, book_title: str) -> list[TextbookSection]:
    sections: list[TextbookSection] = []
    heading_stack: list[str] = []
    current_heading: list[str] = []
    current_lines: list[str] = []
    current_content_type = "textbook_body"
    drop_until_end = False
    has_markdown_headings = any(HEADING_PATTERN.match(line.strip()) for line in text.splitlines())
    body_started = not has_markdown_headings

    def flush() -> None:
        nonlocal body_started, current_lines, current_heading, current_content_type
        body = _strip_noisy_lines("\n".join(current_lines))
        current_lines = []
        if drop_until_end:
            return
        if not current_heading:
            if has_markdown_headings:
                return
            if _is_low_value_body(body):
                return
            sections.append(
                TextbookSection(
                    heading_path=[],
                    body=body,
                    content_type="textbook_body",
                )
            )
            return

        heading = current_heading[-1]
        if (
            _is_noise_heading(heading)
            or _is_catalog_heading(heading)
            or _looks_like_toc_entry(heading, body)
            or _looks_like_page_header(heading, body, book_title)
            or _is_low_value_body(body)
        ):
            return

        if not body_started:
            if not CONTENT_START_PATTERN.match(heading) or _looks_like_catalog_body(body):
                return
            body_started = True
            if any(
                _is_noise_heading(item) or _is_catalog_like_heading(item)
                for item in current_heading[:-1]
            ):
                current_heading = [heading]
        elif any(
            _is_noise_heading(item) or _is_catalog_like_heading(item)
            for item in current_heading[:-1]
        ):
            current_heading = [heading]

        sections.append(
            TextbookSection(
                heading_path=current_heading,
                body=body,
                content_type=current_content_type,
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            flush()
            markdown_level = len(heading_match.group(1))
            heading = _normalize_heading(heading_match.group(2))
            if _is_tail_drop_heading(heading):
                drop_until_end = True
                current_heading = []
                current_lines = []
                continue

            level = _heading_level(markdown_level, heading)
            heading_stack = _update_heading_stack(heading_stack, level, heading)
            current_heading = heading_stack
            current_content_type = "catalog" if _is_catalog_heading(heading) else "textbook_body"
            continue

        if not drop_until_end:
            current_lines.append(raw_line)

    flush()
    return sections


def _fixed_windows(text: str, max_len: int) -> list[str]:
    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        windows.append(text[start:end].strip())
        start = end
    return [window for window in windows if window]


def _split_oversized_unit(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]

    sentence_units = _split_sentences(text)
    if len(sentence_units) > 1:
        result: list[str] = []
        for sentence in sentence_units:
            if len(sentence) <= max_len:
                result.append(sentence)
            else:
                result.extend(_fixed_windows(sentence, max_len))
        return result

    return _fixed_windows(text, max_len)


def _content_units(text: str, max_len: int) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) <= max_len:
            units.append(paragraph)
            continue

        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1:
            for line in lines:
                units.extend(_split_oversized_unit(line, max_len))
            continue

        units.extend(_split_oversized_unit(paragraph, max_len))

    return units


def _merge_units(
    units: list[str],
    *,
    max_len: int,
    overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        separator = "\n\n" if current else ""
        if len(current) + len(separator) + len(unit) <= max_len:
            current = f"{current}{separator}{unit}" if current else unit
            continue

        if current:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail}\n\n{unit}" if tail else unit
        else:
            chunks.append(unit[:max_len].strip())
            current = unit[max_len:].strip()

    if current:
        chunks.append(current)

    return chunks


def _section_to_chunks(
    section: TextbookSection,
    *,
    book_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    prefix = _heading_context(section.heading_path, book_title)
    prefix_text = f"{prefix}\n\n" if prefix else ""
    body_budget = max(160, chunk_size - len(prefix_text))
    units = _content_units(section.body, body_budget)
    merged = _merge_units(units, max_len=body_budget, overlap=chunk_overlap)

    chunks: list[str] = []
    for body in merged:
        content = f"{prefix_text}{body}".strip()
        if not _is_low_value_body(body, min_chars=50):
            chunks.append(content)
    return chunks


def chunk_document(
    document: RagSourceDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagChunk]:
    text = normalize_text(document.content)
    if not text:
        return []

    book_title = document.title
    sections = _iter_sections(text, book_title=book_title)
    if not sections:
        sections = [
            TextbookSection(
                heading_path=[],
                body=text,
                content_type="textbook_body",
            )
        ]

    rag_chunks: list[RagChunk] = []
    for section in sections:
        for content in _section_to_chunks(
            section,
            book_title=book_title,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunk_metadata = dict(document.metadata)
            chunk_metadata.update(
                {
                    "book_title": book_title,
                    "heading_path": section.heading_path,
                    "content_type": section.content_type,
                    "chunk_strategy": "markdown_heading_recursive_v1",
                }
            )
            chunk_index = len(rag_chunks)
            rag_chunks.append(
                RagChunk(
                    chunk_id=_chunk_id(document.doc_id, chunk_index, content),
                    doc_id=document.doc_id,
                    title=document.title,
                    content=content,
                    chunk_index=chunk_index,
                    source_type=document.source_type,
                    source_uri=document.source_uri,
                    language=document.language,
                    tags=document.tags,
                    metadata=chunk_metadata,
                )
            )

    return rag_chunks


def chunk_documents(
    documents: list[RagSourceDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
