from dotenv import load_dotenv
from corpus import load_corpus
from chunkers import naive_chunks, recursive_chunks, sentence_split, semantic_chunks
from embedder import embed_chunks
from retriever import build_index, search, search_window

load_dotenv()

CORPUS_PATH = "data/sample.txt"
TOP_K = 3
WINDOW = 2
SEMANTIC_THRESHOLD = 0.60


def build_all_indexes(text: str) -> dict:
    print("Building indexes (one-time)...")

    naive = naive_chunks(text)
    recursive = recursive_chunks(text)
    sentences = sentence_split(text)
    sent_embeddings = embed_chunks(sentences)
    semantic = semantic_chunks(sentences, sent_embeddings, threshold=SEMANTIC_THRESHOLD)

    indexes = {
        "naive": (naive, build_index(naive, embed_chunks(naive))),
        "recursive": (recursive, build_index(recursive, embed_chunks(recursive))),
        "sentence_window": (sentences, build_index(sentences, sent_embeddings)),
        "semantic": (semantic, build_index(semantic, embed_chunks(semantic))),
    }

    print(f"  naive: {len(naive)} chunks")
    print(f"  recursive: {len(recursive)} chunks")
    print(f"  sentence-window: {len(sentences)} sentences")
    print(f"  semantic: {len(semantic)} chunks  (threshold={SEMANTIC_THRESHOLD})")
    print("\nReady. Type a query (or 'quit' to exit).\n")
    return indexes


def query_all(query: str, indexes: dict) -> None:
    strategies = [
        ("NAIVE", "naive"),
        ("RECURSIVE", "recursive"),
        ("SENTENCE-WINDOW", "sentence_window"),
        ("SEMANTIC", "semantic"),
    ]

    for label, key in strategies:
        chunks, index = indexes[key]
        print(f"\n--- {label} ---")

        if key == "sentence_window":
            results = search_window(query, chunks, index, top_k=1, window=WINDOW)
        else:
            results = search(query, index, top_k=1)

        chunk, score = results[0]
        print(f"score={score:.4f}")
        print(chunk)


def main() -> None:
    text = load_corpus(CORPUS_PATH)
    indexes = build_all_indexes(text)

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() == "quit":
            break
        print(f"\n{'=' * 60}")
        print(f"Query: '{query}'")
        print("=" * 60)
        query_all(query, indexes)
        print()


if __name__ == "__main__":
    main()
