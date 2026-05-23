# Notes

Running log. One entry per working session. Newest at the top.

---

## 2026-05-23 — Scaffold

**What I tried:**
- Grounded project in `concepts/retrieval-augmented-generation` (Level 2) and `sources/100x-cohort7-module2-llm` (L15 Advanced RAG Techniques) from the Zeno wiki.
- Scoped to three chunking strategies: recursive splitting, sentence-window, semantic chunking.
- Same corpus as naive-rag for apples-to-apples comparison.

**Motivation:**
- naive-rag Step 7 identified three root failures: mid-word boundary cuts, pronoun drift (retrieved chunk starts with "It" — subject in previous chunk), vague queries scoring uniformly.
- Each strategy in this project is a direct fix for one or more of those failures.

**Next:**
- Step 1: Reload corpus, re-run naive chunking as baseline, confirm the known failures are reproducible.
