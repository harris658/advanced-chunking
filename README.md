# advanced-chunking

> Three chunking strategies that fix the failure modes of naive fixed-size splitting — benchmarked on the same corpus.

## Showcase

- **What this proves:** I can diagnose chunking failures and benchmark fixes objectively, not by feel.
- **Headline result:** Four strategies on the same corpus + queries — sentence-window won factual lookup (khadi **0.7875** vs naive 0.7193), recursive won broad queries (0.6317); semantic underperformed where domain scores cluster. No single winner.
- **Demo:** `python src/main.py`

## Concepts practiced

- [Chunking](../../concepts/chunking.md)
- [Embeddings & cosine similarity](../../concepts/embeddings-and-cosine-similarity.md)
- [Retrieval-Augmented Generation](../../concepts/retrieval-augmented-generation.md)

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

- [x] Step 1 — Baseline: reload corpus, re-run naive chunking, log the known failures
- [x] Step 2 — Recursive splitting: implement hierarchy splitter, inspect chunk boundaries
- [x] Step 3 — Sentence-window: index sentences, build context window retrieval
- [x] Step 4 — Semantic chunking: embed sentences, detect boundary by similarity drops
- [x] Step 5 — Benchmark: same 4 queries across all strategies, compare retrieved chunks
- [x] Step 6 — Failure analysis: what does each strategy still miss?

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

| Strategy | "what is khadi" | "what is a kurta" | "what can we wear to weddings" |
|---|---|---|---|
| Naive | 0.7193 | 0.6608 | 0.6043 |
| Recursive | 0.7479 | 0.6639 | **0.6317** |
| Sentence-window | **0.7875** | 0.6500 | 0.6309 |
| Semantic | 0.6967 | 0.6530 | 0.5731 |

No single strategy wins every query. Sentence-window is most reliable for specific factual lookups; recursive for broader questions. Semantic chunking requires careful threshold tuning and underperforms on domain-specific corpora where similarity scores cluster in a narrow band.

See `notes.md` for the full session log and failure analysis.
