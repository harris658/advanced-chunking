from dotenv import load_dotenv
from corpus import load_corpus
from chunkers import naive_chunks
from embedder import embed_chunks, embed_query
from retriever import build_index, search

load_dotenv()

CORPUS_PATH = "data/sample.txt"
QUERY = "what is khadi"
TOP_K = 3


def run_baseline() -> None:
    print("=" * 60)
    print("BASELINE — naive fixed-size chunking")
    print("=" * 60)

    text = load_corpus(CORPUS_PATH)
    chunks = naive_chunks(text, size=200, overlap=40)
    print(f"\nCorpus: {len(text)} chars → {len(chunks)} chunks\n")

    for i, c in enumerate(chunks):
        print(f"[{i:02d}] {repr(c)}\n")

    print("-" * 60)
    print(f"Query: '{QUERY}'  (top {TOP_K})")
    print("-" * 60)

    embeddings = embed_chunks(chunks)
    index = build_index(chunks, embeddings)
    results = search(QUERY, index, top_k=TOP_K)

    for rank, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {rank}  score={score:.4f}")
        print(chunk)


if __name__ == "__main__":
    run_baseline()
