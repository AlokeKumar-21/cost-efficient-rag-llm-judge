
import hashlib
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.embeddings import embed_texts
from app.store import VectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}


def extract_text(path: Path) -> str:
    """Extract readable text from PDF, HTML, or Markdown."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix in {".html", ".htm"}:
        return BeautifulSoup(raw, "html.parser").get_text(
            separator="\n", strip=True
        )

    # Markdown is already readable text; retain headings and structure.
    return raw


def split_text(text: str) -> list[str]:
    """Split text into overlapping character-based chunks."""
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def category_for(path: Path, root: Path) -> str:
    """Use the first subfolder as category, or the file stem at root."""
    relative = path.relative_to(root)
    return relative.parts[0] if len(relative.parts) > 1 else path.stem


def ingest_folder(folder: str, batch_size: int = 32) -> dict:
    """Ingest supported files, skipping previously indexed chunk IDs."""
    root = Path(folder).resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Document folder not found: {root}")

    store = VectorStore()
    files_seen = 0
    chunks_seen = 0
    vectors_added = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = extract_text(path)
        chunks = split_text(text)

        if not chunks:
            print(f"Skipping empty document: {path.name}")
            continue

        files_seen += 1
        chunks_seen += len(chunks)
        relative_source = path.relative_to(root).as_posix()
        category = category_for(path, root)

        records = []
        for chunk_index, chunk in enumerate(chunks):
            stable_id = hashlib.sha256(
                f"{relative_source}\0{chunk_index}\0{chunk}".encode("utf-8")
            ).hexdigest()

            records.append({
                "id": stable_id,
                "text": chunk,
                "source": relative_source,
                "filename": path.name,
                "chunk_index": chunk_index,
                "category": category,
            })

        # Generate embeddings in batches to avoid oversized API requests.
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            vectors = embed_texts([item["text"] for item in batch])
            vectors_added += store.add_chunks(vectors, batch)

        print(
            f"Processed {relative_source}: {len(chunks)} chunks",
            flush=True,
        )

    return {
        "files_processed": files_seen,
        "chunks_processed": chunks_seen,
        "vectors_added": vectors_added,
        "total_vectors": store.index.ntotal,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Ingest PDF, HTML, and Markdown documents."
    )
    parser.add_argument("folder", help="Folder containing source documents")
    args = parser.parse_args()

    result = ingest_folder(args.folder)
    print(json.dumps(result, indent=2))
