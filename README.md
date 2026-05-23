# advanced-chunking

> Three chunking strategies that fix the failure modes of naive fixed-size splitting — benchmarked on the same corpus.

## Source

- **Wiki page:** `concepts/retrieval-augmented-generation` — Level 2 Advanced RAG section
- **Course source:** `sources/100x-cohort7-module2-llm` — L15 Advanced RAG Techniques
- **Why this concept:** Naive RAG (previous project) revealed three root failures: mid-word cuts, broken pronoun context, and vague queries producing undifferentiated scores. Advanced chunking directly targets all three.

## Goal

Implement and benchmark three chunking strategies against the naive baseline on the same corpus and query set:

1. Naive (baseline) — fixed 200-char window, 40-char overlap
2. Recursive splitting — hierarchy: `\n\n` → `\n` → `. ` → ` `; respects natural boundaries
3. Sentence-window — embed single sentences; retrieve ±2 surrounding sentences as context
4. Semantic chunking — embed sentences, detect topic boundaries by cosine similarity drops

Concrete success bar: the khadi query that returned a pronoun-only chunk in naive RAG should return a complete, self-contained answer from at least two of the three strategies.

## Approach

- [ ] Step 1 — Baseline: reload corpus, re-run naive chunking, log the known failures
- [ ] Step 2 — Recursive splitting: implement hierarchy splitter, inspect chunk boundaries
- [ ] Step 3 — Sentence-window: index sentences, build context window retrieval
- [ ] Step 4 — Semantic chunking: embed sentences, detect boundary by similarity drops
- [ ] Step 5 — Benchmark: same 4 queries across all strategies, compare retrieved chunks
- [ ] Step 6 — Failure analysis: what does each strategy still miss?

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# .env is symlinked from labs/.env
```

## Run

```bash
python src/main.py
```

## Results

See `notes.md` for the running log.
