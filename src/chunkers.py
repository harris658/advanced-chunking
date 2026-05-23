def naive_chunks(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def recursive_chunks(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= size:
            return [text]

        sep = separators[0]
        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > size:
                    chunks.extend(_split(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw = _split(text, separators)

    # add overlap by prepending tail of previous chunk
    result = []
    for i, chunk in enumerate(raw):
        if i == 0 or not overlap:
            result.append(chunk)
        else:
            prev_tail = result[-1][-overlap:]
            result.append(prev_tail + " | " + chunk)

    return result
