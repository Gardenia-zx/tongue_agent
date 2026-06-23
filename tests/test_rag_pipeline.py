import unittest
from tempfile import TemporaryDirectory
from textwrap import dedent
from pathlib import Path

from app.core.config import get_settings
from app.rag.chunker import chunk_document, chunk_documents, normalize_text
from app.rag.document_loader import load_documents, load_seed_documents
from app.rag.query_expansion import expand_rag_query
from app.schemas.rag import RagSourceDocument


class TestRagPipeline(unittest.TestCase):
    def test_seed_documents_can_be_chunked(self) -> None:
        settings = get_settings()
        documents = load_seed_documents()
        chunks = chunk_documents(
            documents,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

        self.assertGreaterEqual(len(documents), 5)
        self.assertGreaterEqual(len(chunks), len(documents))

        chunk_ids = {chunk.chunk_id for chunk in chunks}
        self.assertEqual(len(chunk_ids), len(chunks))

        for chunk in chunks:
            self.assertTrue(chunk.content.strip())
            self.assertLessEqual(len(chunk.content), settings.rag_chunk_size + 120)
            self.assertTrue(chunk.title)
            self.assertTrue(chunk.doc_id)

    def test_normalize_text_collapses_spaces_and_blank_lines(self) -> None:
        raw_text = " 舌象   观察\r\n\r\n\r\n需要自然光。 "
        normalized = normalize_text(raw_text)

        self.assertEqual("舌象 观察\n\n需要自然光。", normalized)

    def test_local_documents_include_source_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "诊断" / "中医诊断学.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("湿气重是常见健康表达。", encoding="utf-8")

            documents = load_documents(
                directory=tmp_dir,
                source_repo="https://github.com/PanckooAI/TCM_Datasets",
                source_commit="abc123",
                license_name="source-license",
            )

        self.assertEqual(1, len(documents))
        document = documents[0]
        self.assertEqual("中医诊断学", document.title)
        self.assertEqual("诊断/中医诊断学.md", document.source_uri)
        self.assertEqual(
            "https://github.com/PanckooAI/TCM_Datasets",
            document.metadata["source_repo"],
        )
        self.assertEqual("abc123", document.metadata["source_commit"])
        self.assertEqual("source-license", document.metadata["license"])
        self.assertEqual("local_file", document.metadata["content_storage"])

    def test_textbook_chunker_filters_front_matter_catalog_and_tail_index(self) -> None:
        document = RagSourceDocument(
            doc_id="doc_textbook",
            title="中医诊断学",
            content=dedent("""
![](images/cover.jpg)

# 全国高等中医药院校规划教材（第十一版）
<html><body><table><tr><td>中医基础理论</td><td>中医诊断学</td></tr></table></body></html>

# 主编
张三 李四

# 目录

# 第一章 舌诊基础 1
第一节 舌象观察 1
第二节 舌苔观察 8

# 第一章 舌诊基础

# 第一节 舌象观察

舌象观察需要自然光，拍摄时应避免滤镜、强反光和明显偏色。

观察舌质、舌苔、津液和舌体形态时，需要结合整体状态进行一般健康知识解释，不能仅凭单一表现作出诊断。

# 索引
舌苔 99
"""),
            source_type="local_file",
            source_uri="中医诊断学.md",
        )

        chunks = chunk_document(document, chunk_size=260, chunk_overlap=40)
        joined = "\n".join(chunk.content for chunk in chunks)

        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("舌象观察需要自然光", joined)
        self.assertNotIn("cover.jpg", joined)
        self.assertNotIn("<table", joined)
        self.assertNotIn("主编", joined)
        self.assertNotIn("目录", joined)
        self.assertNotIn("索引", joined)

        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 260)
            self.assertEqual("markdown_heading_recursive_v1", chunk.metadata["chunk_strategy"])
            self.assertEqual("textbook_body", chunk.metadata["content_type"])
            self.assertIn("第一节 舌象观察", chunk.metadata["heading_path"])

    def test_rag_query_expands_colloquial_dampness_terms(self) -> None:
        expanded = expand_rag_query("湿气重是什么意思")

        self.assertIn("湿气重是什么意思", expanded.search_text)
        self.assertIn("湿邪", expanded.terms)
        self.assertIn("痰湿", expanded.terms)
        self.assertIn("舌苔厚腻", expanded.terms)
