from pathlib import Path

from app.rag.seed_corpus import SEED_DOCUMENTS
from app.schemas.rag import RagSourceDocument


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_seed_documents() -> list[RagSourceDocument]:
    return list(SEED_DOCUMENTS)


def load_documents_from_directory(
    directory: str | Path,
    *,
    source_repo: str | None = None,
    source_commit: str | None = None,
    license_name: str | None = None,
) -> list[RagSourceDocument]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Knowledge directory does not exist: {root}")

    documents: list[RagSourceDocument] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = path.relative_to(root).as_posix()
        doc_id = "file_" + relative_path.replace("/", "_").replace(".", "_")
        documents.append(
            RagSourceDocument(
                doc_id=doc_id,
                title=path.stem,
                content=content,
                source_type="local_file",
                source_uri=relative_path,
                tags=[path.parent.name] if path.parent != root else [],
                metadata={
                    "file_name": path.name,
                    "source_repo": source_repo,
                    "source_path": relative_path,
                    "source_commit": source_commit,
                    "license": license_name,
                    "content_storage": "local_file",
                    "content_path": str(path),
                },
            )
        )

    return documents


def load_documents(
    *,
    directory: str | Path | None = None,
    source_repo: str | None = None,
    source_commit: str | None = None,
    license_name: str | None = None,
) -> list[RagSourceDocument]:
    if directory is None:
        return load_seed_documents()

    documents = load_documents_from_directory(
        directory,
        source_repo=source_repo,
        source_commit=source_commit,
        license_name=license_name,
    )
    if not documents:
        raise ValueError(f"No supported knowledge documents found in: {directory}")

    return documents
