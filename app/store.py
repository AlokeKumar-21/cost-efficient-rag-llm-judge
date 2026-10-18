
import json
from pathlib import Path

import faiss
import numpy as np

from app.config import INDEX_PATH, METADATA_PATH, EMBEDDING_DIMENSION


class VectorStore:
    """Persistent FAISS index with JSON metadata and cosine similarity."""

    def __init__(self):
        self.index_path = Path(INDEX_PATH)
        self.metadata_path = Path(METADATA_PATH)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            if self.index.d != EMBEDDING_DIMENSION:
                raise ValueError("Stored index has an unexpected dimension.")
            if self.index.ntotal != len(self.metadata):
                raise ValueError("Index and metadata record counts differ.")
        else:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            self.metadata = []

    def save(self):
        """Persist index and metadata."""
        faiss.write_index(self.index, str(self.index_path))
        temp_path = self.metadata_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.metadata_path)

    def add_chunks(self, vectors: np.ndarray, chunks: list[dict]):
        """Add new chunks, skipping IDs already present."""
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.shape != (len(chunks), EMBEDDING_DIMENSION):
            raise ValueError("Vector dimensions do not match chunk records.")

        existing_ids = {item["id"] for item in self.metadata}
        new_vectors = []
        new_chunks = []

        for vector, chunk in zip(vectors, chunks):
            if chunk["id"] not in existing_ids:
                new_vectors.append(vector)
                new_chunks.append(chunk)
                existing_ids.add(chunk["id"])

        if new_vectors:
            matrix = np.stack(new_vectors).astype(np.float32)
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
            self.metadata.extend(new_chunks)
            self.save()

        return len(new_chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[dict]:
        """Search by cosine similarity, optionally filtering by category."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError("Query vector has an unexpected dimension.")

        faiss.normalize_L2(query)
        candidate_count = self.index.ntotal
        scores, positions = self.index.search(query, candidate_count)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue

            item = self.metadata[int(position)]
            if category and item.get("category") != category:
                continue

            results.append({**item, "score": float(score)})
            if len(results) >= top_k:
                break

        return results
