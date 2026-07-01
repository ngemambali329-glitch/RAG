"""
knowledge_base.py
-------------------
Shared document-ingestion logic used by both the CLI ingest script and the
Streamlit web UI's built-in file uploader. Loads PDFs/TXT/MD, chunks them,
embeds them locally, and stores/updates them in the persistent Chroma DB.
"""
import hashlib
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

import config


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


SUPPORTED_EXTENSIONS = {".pdf": read_pdf, ".txt": read_text_file, ".md": read_text_file}


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """Return list of (filename, raw_text) for every supported file in data_dir."""
    docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
            try:
                text = SUPPORTED_EXTENSIONS[path.suffix.lower()](path)
                if text.strip():
                    docs.append((path.name, text))
            except Exception:
                pass
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunk_id(filename: str, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{filename}-{idx}-{text[:50]}".encode()).hexdigest()[:12]
    return f"{filename}-{idx}-{h}"


def get_collection():
    """Get (or create) the Chroma collection using the configured embedding model."""
    chroma_client = chromadb.PersistentClient(path=str(config.DB_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    return chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_file(filename: str, text: str, collection=None) -> int:
    """Chunk + embed + store a single document. Returns number of chunks stored."""
    if collection is None:
        collection = get_collection()
    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return 0
    ids = [build_chunk_id(filename, i, c) for i, c in enumerate(chunks)]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


def ingest_all(data_dir: Path = None) -> dict:
    """Ingest every supported file currently in data_dir. Returns a summary dict."""
    data_dir = data_dir or config.DATA_DIR
    collection = get_collection()
    docs = load_documents(data_dir)
    total_chunks = 0
    per_file = {}
    for filename, text in docs:
        n = ingest_file(filename, text, collection=collection)
        per_file[filename] = n
        total_chunks += n
    return {"files_processed": len(docs), "total_chunks": total_chunks, "per_file": per_file}


def list_indexed_sources() -> List[str]:
    """Return the distinct source filenames currently stored in the knowledge base."""
    try:
        collection = get_collection()
        data = collection.get(include=["metadatas"])
        sources = {m.get("source") for m in data.get("metadatas", []) if m}
        return sorted(s for s in sources if s)
    except Exception:
        return []


def collection_chunk_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0
