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


def search_window(
    query: str,
    sentences: list[str],
    index: list[dict],
    top_k: int = 3,
    window: int = 2,
) -> list[tuple[str, float]]:
    from embedder import embed_query

    q_vec = embed_query(query)
    scored = [(i, cosine_similarity(q_vec, entry["embedding"])) for i, entry in enumerate(index)]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scored[:top_k]:
        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        context = " ".join(sentences[start:end])
        results.append((context, score))

    return results
