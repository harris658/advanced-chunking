# Notes

Running log. One entry per working session. Newest at the top.

---

## 2026-05-23 — Steps 1–6: Full session

### Step 1 — Baseline confirmed
Re-ran naive fixed-size chunking (200 chars, 40 overlap) on the same corpus.
- Result 1 for "what is khadi": score 0.7193, starts with `"It was popularised..."` — pronoun drift confirmed.
- Results 2–3 scored 0.46–0.47 — generic menswear vocabulary, not relevance.
- Mid-word cuts visible throughout (`"woven in Va"`, `"It typical"`).

### Step 2 — Recursive splitting
Hierarchy splitter: `\n\n` → `\n` → `. ` → ` `. Never cuts mid-word.
- 14 chunks vs 12.
- Chunk [04] now contains `"Khadi is a hand-spun, hand-woven cloth central to Indian identity"` — definition intact.
- Result 2 jumped from 0.4665 → 0.7318 — pronoun drift partially fixed.
- Still cuts mid-sentence in some places when sentence exceeds size limit.
- Overlap implemented as `prev_tail + " | " + chunk` — visible separator is a crutch.

### Step 3 — Sentence-window
Index: 20 individual sentences. Retrieval: matched sentence ± 2 surrounding sentences.
- Result 1 for "what is khadi": score 0.7875 — best across all strategies for this query.
- Opens with `"Khadi is a hand-spun, hand-woven cloth central to Indian identity."` — zero pronoun drift.
- Window=2 overreaches slightly: Result 1 bleeds into Banarasi silk (next topic, 2 sentences forward).
- Redundant retrieval: Results 2 and 3 were near-identical windows from the same opening paragraph.

### Step 4 — Semantic chunking
Embed each sentence, split where adjacent similarity < threshold.
- Threshold=0.5 → 1 chunk (entire corpus). All menswear sentences score above 0.5 — domain vocabulary compresses the signal.
- Printed similarity table: scores ranged 0.52–0.80 with no extreme outlier. Real dips at [07→08] 0.5729, [15→16] 0.5226, [17→18] 0.5575, [18→19] 0.5619.
- Threshold=0.60 → 5 chunks. But split *within* the khadi section — definition and usage landed in different chunks.
- Result 1 for "what is khadi": score 0.6967 — worse than both recursive and sentence-window.
- Chunks wildly uneven: one huge chunk (khadi + Banarasi + zari + block printing merged), several single-sentence chunks.

### Step 5 — Interactive benchmark
Queries: "what is khadi", "what is a kurta", "what can we wear to weddings?"

| Query | Best strategy | Score |
|-------|--------------|-------|
| what is khadi | sentence-window | 0.7875 |
| what is a kurta | recursive | 0.6639 |
| what can we wear to weddings | recursive | 0.6317 |

No single strategy won every query.

### Step 6 — Failure analysis

**What each strategy still misses:**

**Naive**
- Mid-word and mid-sentence cuts are unfixable — it's a character counter with no awareness of meaning.
- Score can look reasonable (0.60+) even when the retrieved text starts mid-word and is unusable.

**Recursive**
- Fixes boundary cuts but chunk size is unpredictable — one chunk might be 1 sentence, another 4.
- The `" | "` overlap is visible in output and pollutes the retrieved text.
- No awareness of topic — will merge unrelated sentences if they fit within the size limit.

**Sentence-window**
- Window size is a fixed tuning knob. Window=2 overshoots on topic transitions; window=1 may miss context.
- Redundant retrieval: multiple sentences in the same paragraph all return overlapping windows. Top-3 results can be near-identical.
- Scores are diluted by window width — a 5-sentence window pulls the vector in multiple directions, lowering the similarity score even when the content is directly relevant.

**Semantic chunking**
- Entirely dependent on threshold — no principled way to pick it without inspecting the similarity distribution first.
- On domain-specific corpora, similarity scores cluster in a narrow band (0.52–0.80 here). Hard to find clean cut points.
- Can split within a single topic if two adjacent sentences happen to use different vocabulary.
- Produces uneven chunk sizes — single-sentence chunks alongside multi-paragraph chunks.

**Root causes that none of these fix:**
1. **No re-ranking** — retrieval score is the only signal. A chunk that scores 0.60 and directly answers the question ranks below one that scores 0.65 but answers a different question.
2. **Single-vector per chunk** — one embedding represents the whole chunk. Long chunks average out the meaning; short chunks lose context.
3. **Query-document vocabulary mismatch** — all four strategies use dense embeddings only. If the query uses different words than the document (synonyms, paraphrases), similarity drops even when the meaning matches. Hybrid search (BM25 + vector) addresses this.

**What comes next in Advanced RAG:**
- Re-ranking: retrieve top-20 by vector similarity, re-rank by a cross-encoder model that reads query + chunk together.
- Hybrid search: combine BM25 (keyword) + vector (semantic) scores. BM25 catches exact vocabulary matches that vector misses.
- Query expansion: rewrite the query multiple ways before retrieval to improve recall.
