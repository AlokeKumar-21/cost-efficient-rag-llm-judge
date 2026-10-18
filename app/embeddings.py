
import numpy as np
from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Set it in your .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate normalized float32 embeddings for text inputs."""
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    vectors = [item.values for item in response.embeddings]

    if len(vectors) != len(texts):
        raise RuntimeError("Embedding response count does not match input.")

    array = np.asarray(vectors, dtype=np.float32)

    if array.shape != (len(texts), EMBEDDING_DIMENSION):
        raise RuntimeError(f"Unexpected embedding shape: {array.shape}")

    # Normalize so FAISS inner-product search corresponds to cosine similarity.
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Received a zero-length embedding vector.")

    return array / norms
