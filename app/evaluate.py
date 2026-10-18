
import json
import math
import time
from pathlib import Path

from app.config import TOP_K
from app.embeddings import embed_texts
from app.store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation_questions.json"
OUTPUT_PATH = BASE_DIR / "data" / "evaluation_results.json"

K = TOP_K


def recall_at_k(retrieved_ids, relevant_ids):
    """Fraction of relevant chunks retrieved in the top-k results."""
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return None

    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids, relevant_ids):
    """Return 1 if at least one relevant chunk is retrieved; otherwise 0."""
    if not relevant_ids:
        return None

    return int(bool(set(retrieved_ids) & set(relevant_ids)))


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    relevant_ids = set(relevant_ids)

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids):
    """Normalized discounted cumulative gain for binary relevance."""
    relevant_ids = set(relevant_ids)

    if not relevant_ids:
        return None

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(retrieved_ids, start=1)
        if item_id in relevant_ids
    )

    ideal_count = min(len(relevant_ids), len(retrieved_ids))

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0:
        return 0.0

    return min(dcg / idcg, 1.0)


def context_precision_at_k(retrieved_ids, relevant_ids):
    """
    Average precision among retrieved chunks.
    This is a provisional estimate because relevance labels
    are inferred from keyword matches.
    """
    relevant_ids = set(relevant_ids)

    if not retrieved_ids or not relevant_ids:
        return None

    relevant_seen = 0
    precision_sum = 0.0

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            relevant_seen += 1
            precision_sum += relevant_seen / rank

    if relevant_seen == 0:
        return 0.0

    return precision_sum / relevant_seen


def percentile(values, p):
    """Nearest-rank percentile."""
    if not values:
        return None

    sorted_values = sorted(values)
    index = max(0, math.ceil(p * len(sorted_values)) - 1)
    return sorted_values[index]


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    questions = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    store = VectorStore()

    if store.index.ntotal == 0:
        raise RuntimeError(
            "No indexed chunks. Run document ingestion first."
        )

    rows = []
    retrieval_times = []

    for item in questions:
        question = item["question"]

        relevant_terms = [
            term.lower()
            for term in item.get("relevant_terms", [])
            if term.strip()
        ]

        # Generate the query embedding outside the FAISS timer.
        query_vector = embed_texts([question])[0]

        start = time.perf_counter()
        results = store.search(query_vector, top_k=K)
        retrieval_ms = (time.perf_counter() - start) * 1000

        retrieval_times.append(retrieval_ms)

        retrieved_ids = [
            f'{result["source"]}::{result["chunk_index"]}'
            for result in results
        ]

        # Provisional relevance labels based on keyword matching.
        # These are not manually validated ground-truth labels.
        relevant_ids = set()

        if relevant_terms:
            for metadata in store.metadata:
                chunk_text = metadata.get("text", "").lower()

                if any(term in chunk_text for term in relevant_terms):
                    relevant_ids.add(
                        f'{metadata["source"]}::{metadata["chunk_index"]}'
                    )

        row = {
            "id": item.get("id"),
            "question": question,
            "retrieval_ms": round(retrieval_ms, 3),
            "retrieved_chunks": len(results),
            "relevant_chunks_in_index": len(relevant_ids),
            "recall_at_k": recall_at_k(
                retrieved_ids, relevant_ids
            ),
            "hit_rate_at_k": hit_rate_at_k(
                retrieved_ids, relevant_ids
            ),
            "mrr": reciprocal_rank(
                retrieved_ids, relevant_ids
            ),
            "ndcg_at_k": ndcg_at_k(
                retrieved_ids, relevant_ids
            ),
            "context_precision_at_k": context_precision_at_k(
                retrieved_ids, relevant_ids
            ),
            "retrieved_sources": [
                result["source"] for result in results
            ],
            "retrieved_chunk_ids": retrieved_ids,
        }

        rows.append(row)

    def average(key):
        values = [
            row[key]
            for row in rows
            if row[key] is not None
        ]

        return sum(values) / len(values) if values else None

    report = {
        "questions_evaluated": len(rows),
        "k": K,
        "mean_recall_at_k": average("recall_at_k"),
        "mean_hit_rate_at_k": average("hit_rate_at_k"),
        "mean_mrr": average("mrr"),
        "mean_ndcg_at_k": average("ndcg_at_k"),
        "mean_context_precision_at_k": average(
            "context_precision_at_k"
        ),
        "retrieval_latency_p50_ms": percentile(
            retrieval_times, 0.50
        ),
        "retrieval_latency_p95_ms": percentile(
            retrieval_times, 0.95
        ),
        "results": rows,
        "limitations": [
            (
                "Relevance labels are inferred from keyword matches "
                "and are not manually validated."
            ),
            (
                "Questions without relevant_terms are excluded from "
                "Recall@k, Hit Rate@k, nDCG@k, and context precision averages."
            ),
            (
                "Context precision is a provisional estimate based on "
                "keyword-derived relevance labels."
            ),
            (
                "Latency measures local FAISS search only. It excludes "
                "embedding API latency, network time, and answer generation."
            ),
            (
                "The evaluation dataset and indexed document collection "
                "are small; results should not be treated as generalizable."
            ),
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        key: value
        for key, value in report.items()
        if key != "results"
    }

    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
