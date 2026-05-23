def load_corpus(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
