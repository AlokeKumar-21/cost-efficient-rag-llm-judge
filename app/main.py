
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer_question
from app.store import VectorStore

app = FastAPI(
    title="Cost-Efficient RAG API",
    description="A document-grounded question-answering API.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


@app.get("/health")
def health():
    try:
        store = VectorStore()
        return {
            "status": "ok",
            "indexed_chunks": store.index.ntotal,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store unavailable: {exc}",
        )


@app.post("/query")
def query(request: QueryRequest):
    try:
        return answer_question(
            question=request.question.strip(),
            top_k=request.top_k,
            category=request.category,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the query.",
        )
