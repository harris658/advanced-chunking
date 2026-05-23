from dotenv import load_dotenv
from corpus import load_corpus
from chunkers import naive_chunks, recursive_chunks, sentence_split, semantic_chunks
from embedder import embed_chunks
from retriever import build_index, search, search_window

load_dotenv()

CORPUS_PATH = "data/sample.txt"
QUERY = "what is khadi"
TOP_K = 3


def run_strategy(label: str, chunks: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}  ({len(chunks)} chunks)")
    print("=" * 60)

    print("\n--- Chunk boundaries (first 6) ---")
    for i, c in enumerate(chunks[:6]):
        print(f"[{i:02d}] {repr(c)}\n")

    print(f"--- Query: '{QUERY}'  top {TOP_K} ---")
    embeddings = embed_chunks(chunks)
    index = build_index(chunks, embeddings)
    results = search(QUERY, index, top_k=TOP_K)

    for rank, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {rank}  score={score:.4f}")
        print(chunk)


def run_sentence_window(sentences: list[str], embeddings: list) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP 3 — sentence-window  ({len(sentences)} sentences, window=±2)")
    print("=" * 60)

    index = build_index(sentences, embeddings)
    print(f"--- Query: '{QUERY}'  top {TOP_K} ---")
    results = search_window(QUERY, sentences, index, top_k=TOP_K, window=2)

    for rank, (context, score) in enumerate(results, 1):
        print(f"\nResult {rank}  matched-sentence score={score:.4f}")
        print(context)


def run_semantic(sentences: list[str], embeddings: list, threshold: float = 0.5) -> None:
    from retriever import cosine_similarity
    print(f"\n{'=' * 60}")
    print("STEP 4 — adjacent sentence similarity scores")
    print("=" * 60)
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        marker = " <-- boundary?" if sim < threshold else ""
        print(f"  [{i-1:02d}→{i:02d}]  {sim:.4f}{marker}")

    chunks = semantic_chunks(sentences, embeddings, threshold=threshold)
    print(f"\n{'=' * 60}")
    print(f"STEP 4 — semantic chunking  ({len(chunks)} chunks, threshold={threshold})")
    print("=" * 60)

    print("\n--- Semantic chunks ---")
    for i, c in enumerate(chunks):
        print(f"[{i:02d}] {c}\n")

    print(f"--- Query: '{QUERY}'  top {TOP_K} ---")
    chunk_embeddings = embed_chunks(chunks)
    index = build_index(chunks, chunk_embeddings)
    results = search(QUERY, index, top_k=TOP_K)

    for rank, (chunk, score) in enumerate(results, 1):
        print(f"\nResult {rank}  score={score:.4f}")
        print(chunk)


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    print(f"Corpus: {len(text)} chars\n")

    run_strategy("BASELINE — naive fixed-size", naive_chunks(text))
    run_strategy("STEP 2 — recursive splitting", recursive_chunks(text))

    sentences = sentence_split(text)
    sent_embeddings = embed_chunks(sentences)

    run_sentence_window(sentences, sent_embeddings)
    run_semantic(sentences, sent_embeddings, threshold=0.60)


if __name__ == "__main__":
    main()
