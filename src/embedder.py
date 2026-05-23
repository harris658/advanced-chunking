import numpy as np
import ollama

MODEL = "nomic-embed-text"


def embed_chunks(chunks: list[str]) -> list[np.ndarray]:
    return [
        np.array(ollama.embeddings(model=MODEL, prompt=c).embedding)
        for c in chunks
    ]


def embed_query(query: str) -> np.ndarray:
    return np.array(ollama.embeddings(model=MODEL, prompt=query).embedding)
