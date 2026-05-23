from dotenv import load_dotenv
from corpus import load_corpus
from chunkers import naive_chunks, recursive_chunks, sentence_split
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


def run_sentence_window(text: str) -> None:
    sentences = sentence_split(text)
    print(f"\n{'=' * 60}")
    print(f"STEP 3 — sentence-window  ({len(sentences)} sentences, window=±2)")
    print("=" * 60)

    print("\n--- Individual sentences (first 6) ---")
    for i, s in enumerate(sentences[:6]):
        print(f"[{i:02d}] {s}\n")

    print(f"--- Query: '{QUERY}'  top {TOP_K} ---")
    embeddings = embed_chunks(sentences)
    index = build_index(sentences, embeddings)
    results = search_window(QUERY, sentences, index, top_k=TOP_K, window=2)

    for rank, (context, score) in enumerate(results, 1):
        print(f"\nResult {rank}  matched-sentence score={score:.4f}")
        print(context)


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    print(f"Corpus: {len(text)} chars\n")

    run_strategy("BASELINE — naive fixed-size", naive_chunks(text))
    run_strategy("STEP 2 — recursive splitting", recursive_chunks(text))
    run_sentence_window(text)


if __name__ == "__main__":
    main()
