import numpy as np


def build_index(chunks: list[str], embeddings: list[np.ndarray]) -> list[dict]:
    return [{"text": c, "embedding": e} for c, e in zip(chunks, embeddings)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(
    query: str,
    index: list[dict],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    from embedder import embed_query

    q_vec = embed_query(query)
    scored = [(entry["text"], cosine_similarity(q_vec, entry["embedding"])) for entry in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
