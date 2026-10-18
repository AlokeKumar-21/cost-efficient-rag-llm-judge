
import logging
import time

from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from app.embeddings import embed_texts
from app.store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

client = genai.Client(api_key=GEMINI_API_KEY)

# Tune this using the evaluation set; it is an initial heuristic,
# not a validated relevance threshold.
MIN_SIMILARITY = 0.55


def answer_question(
    question: str,
    top_k: int = TOP_K,
    category: str | None = None,
) -> dict:
    """Retrieve evidence and generate a cited answer grounded in that evidence."""
    started = time.perf_counter()

    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    store = VectorStore()

    if store.index.ntotal == 0:
        return {
            "answer": "I don't have enough relevant information in the indexed documents to answer that question.",
            "citations": [],
            "retrieved_chunks": 0,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "status": "no_context",
        }

    query_vector = embed_texts([question])[0]
    candidates = store.search(query_vector, top_k=top_k, category=category)

    relevant = [
        item for item in candidates
        if item["score"] >= MIN_SIMILARITY
    ]

    if not relevant:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "status=no_context latency_ms=%.2f retrieved_chunks=0",
            elapsed,
        )
        return {
            "answer": "I don't have enough relevant information in the indexed documents to answer that question.",
            "citations": [],
            "retrieved_chunks": 0,
            "latency_ms": elapsed,
            "status": "no_context",
        }

    evidence_blocks = []
    citations = []

    for number, item in enumerate(relevant, start=1):
        evidence_blocks.append(
            f"[{number}] Source: {item['source']}, "
            f"chunk: {item['chunk_index']}\n{item['text']}"
        )
        citations.append({
            "citation_id": number,
            "source": item["source"],
            "chunk_index": item["chunk_index"],
            "similarity": round(item["score"], 4),
        })

    prompt = f"""
You are a careful document question-answering assistant.

Answer the question using ONLY the evidence below.
Do not use outside knowledge or invent missing facts.
If the evidence does not answer the question, explicitly say so.
Cite factual claims using the evidence IDs, such as [1] or [2].
Do not cite an ID that does not exist in the evidence.

EVIDENCE:
{chr(10).join(evidence_blocks)}

QUESTION:
{question}

Give a concise answer with citations.
""".strip()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    answer = (response.text or "").strip()
    if not answer:
        answer = (
            "I couldn't generate a reliable answer from the retrieved evidence."
        )

    elapsed = round((time.perf_counter() - started) * 1000, 2)

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)

    logger.info(
        "status=answered latency_ms=%.2f retrieved_chunks=%d "
        "prompt_tokens=%s output_tokens=%s",
        elapsed,
        len(relevant),
        prompt_tokens,
        output_tokens,
    )

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": len(relevant),
        "latency_ms": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "status": "answered",
    }
