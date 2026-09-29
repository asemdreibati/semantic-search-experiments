# Semantic search over the 7 documents, with "no relevant info" detection

Goal: answer a question from the 7 PDFs when one of them covers it, and
reply **"لا توجد معلومات ذات صلة في الوثائق"** when none does.

## Result

The pipeline here runs end-to-end on the 7 PDFs and is evaluated on 73
questions: 57 answerable and 16 unanswerable, of which 12 are hard
negatives like "حد السرعة على الطرق السريعة" or "عقوبة تسريب البيانات الشخصية".

| | Majd's run (live API) | e5 cosine threshold | **e5 + reranker** |
|---|---|---|---|
| correct document ranked #1 | 0 / 33 counted (see below) | 57 / 57 (100%) | **57 / 57 (100%)** |
| expected evidence in top-5 chunks | – | 93.0% | **96.5%** |
| answerable → answered from right doc (cross-validated) | – | 91.2% | **98.2%** |
| unanswerable → "no relevant info" (cross-validated) | 4 / 4 "pass" (not real, see below) | 93.8% | **93.8%** |
| **overall (cross-validated)** | 10.8% | 91.8% | **97.3%** |
| separation of answerable vs unanswerable (AUC) | – | 0.973 | **0.995** |

Per-query details: [`results_rerank.md`](results_rerank.md) (with reranker),
[`results_title_prefix.md`](results_title_prefix.md) (cosine only).

## Why Majd's experiment failed

Findings from `QA result.rb`, the QA script and the Elastic query:

1. **5 of the 7 PDFs are scanned images with no text layer.** These are
   مجلس الوزراء, حوكمة التنسيق, هيئة الطرق, السلامة الإسعافية and لائحة
   الاتصالات. If ingestion doesn't OCR them, their embeddings are empty
   or garbage and they can't be found. The other 2 have broken Arabic text
   layers: presentation-form glyphs (`ﻣﺪﻳﺮ`) and a reversed lam-alef
   (`اإلصدار`). The titles of the 3 other tested documents appear in none
   of the 185 returned hits. `لائحة الاتصالات` was found, so check in
   Elastic what text was actually indexed for each document.
2. **The expected UUIDs are stale.** None of the 4 UUIDs in the script
   appears anywhere in the results. For example, `لائحة الاتصالات الرسمية`
   is indexed as `220df768-…` and was ranked #1 for 7 of its 10 questions,
   but every one of those was counted as FAIL.
3. **The scores are rank-only.** Every top hit has score `0.016393… = 1/61`,
   the next `1/62`, and so on. That is reciprocal-rank fusion: the score
   says which position a hit is in, not how similar it is. A coffee recipe
   and an exact match both score 1/61, so **no threshold on these scores
   can ever say "no relevant info"**.
4. **The Elastic `min_score: 0.7` filters nothing.** For a cosine `knn`
   query Elastic scores `(1 + cos) / 2`, so 0.7 means cos ≥ 0.4. Modern
   embedding models put *every* Arabic sentence pair above that. The
   calibrated cut-off here is cos ≈ 0.82, which is ≈ 0.91 in Elastic score.
5. **The negative tests could not fail.** A negative "passed" when the top
   hit was not one of the 4 stale UUIDs, which was always true. In fact
   every garbage query returned 8 hits; "car engine overheating" ranked
   `لائحة الاتصالات الرسمية` #1.
6. **The index is mixed with ~80 test correspondences** (`test2`,
   `outgoing - AI - 6`, …) that compete with the 7 documents. Only 4 of
   the 7 documents were tested at all.

## The method (what to do)

```
PDF ──OCR (ara)──► normalise ──► chunk ~800 chars + doc title ──► e5 "passage: " ──► index
question ──normalise──► e5 "query: " ──► cosine top-20 ──► bge-reranker-v2-m3 reads (question, chunk)
         ──► best reranker score ≥ T ? answer from that chunk : "لا توجد معلومات"
```

1. **Extract text properly.** OCR every page with Tesseract `ara` at 300 dpi.
   `ara+eng` was tested and is worse: Arabic words get misread as Latin.
   Normalise documents and queries the same way: NFKC, then strip tashkeel,
   tatweel and bidi marks (`textnorm.py`).
2. **Chunk with context.** Use chunks of about 800 characters with one
   paragraph of overlap, each prefixed with its document title.
3. **Use the embedding model the way it was trained.**
   `multilingual-e5-large` (1024-d, same size as the vectors in the Nuxeo
   query) needs the `query: ` and `passage: ` prefixes.
4. **Use e5 for retrieval, not for the final decision.** A threshold on
   the raw cosine of the best chunk reaches 91.8%. Pick the threshold on labelled questions, including hard
   negatives, and measure it with cross-validation, never by guessing.
   Relative scores such as z-score or top-1 minus mean were tested and
   separate worse (AUC 0.74–0.91 vs 0.97).
5. **Decide with a cross-encoder reranker (`rerank.py`).**
   `BAAI/bge-reranker-v2-m3` reads the question and each of the e5 top-20
   chunks *together* and outputs a relevance score with a real zero.

   | best reranker score | min | median | max |
   |---|---|---|---|
   | answerable (57) | 0.004 | **0.901** | 0.999 |
   | hard negatives (12) | 0.000 | **0.003** | 0.013 |
   | garbage (4) | 0.000 | **0.000** | 0.000 |

   Garbage now scores exactly 0 (cosine: 0.78). Overall accuracy rises
   from 91.8% to **97.3%**, and AUC from 0.973 to **0.995**. Caveats:
   - **The threshold is low (≈0.013) and the margin is thin.** The highest
     unanswerable question scores 0.013, and the lowest correctly
     answered one 0.014. 16 negatives is too few to trust a threshold
     that low, so add more hard negatives to `testset.json` before fixing
     it for production.
   - **Its one remaining error** is "ماذا يحدث عند حدوث عطل تقني أثناء
     الحصة الافتراضية" (0.004). The answer exists but is worded
     differently and sits in a long chunk. Low-scoring answerable
     questions tend to be chunks that mix several articles with OCR
     letterhead noise ("المملكة العربية السعودية … الرقم"). Stripping
     repeated headers and using shorter chunks are the next things to try.
   - **Speed:** about 20 s per question on 4 CPU cores (20 pairs, XLM-R
     large). Production needs a GPU (tens of ms), a smaller K, or an ONNX
     or int8 export.
   - An LLM that answers only from the retrieved chunks, and says "لا توجد
     معلومات ذات صلة في الوثائق" otherwise, is a further safety net. It
     has not been tested here.

### Why every cosine is ~0.8, and what removing that does (`calibration.py`)

The e5 vectors are **anisotropic**: the mean of all chunk vectors has
norm 0.91, so ~91% of every vector is one shared direction. Two chunks
from unrelated documents have a median cosine of 0.81, and "طريقة تحضير
القهوة العربية" scores 0.78 against the regulations.

Subtracting each space's mean before the cosine fixes the *scale*. The
query mean comes from 40 unrelated background questions, not the test
set.

| best-chunk score | raw cosine | centered |
|---|---|---|
| answerable (57) | 0.78 – 0.90 | 0.16 – **0.35** (median) – 0.58 |
| hard negatives (12) | 0.77 – 0.82 | 0.12 – **0.19** – 0.25 |
| garbage (4) | 0.74 – 0.79 | 0.11 – **0.11** – 0.20 |

It does **not** fix the *separation*. AUC is 0.977 centered vs 0.973
raw, and CV accuracy is 89.0% vs 91.8%. Dropping principal components,
whitening, BM25, and BM25 combined with the cosine all do worse (see
`calibration.md`). The questions still mis-decided are the same ones.
They are not caused by the high number but by the bi-encoder itself: one
vector per chunk measures *topic*, not *"does this passage answer this
question"*. "حد السرعة على الطرق السريعة" is on-topic for the roads
regulation but unanswered. "عطل تقني أثناء الحصة الافتراضية" is answered
in different words ("المشكلات التقنية الطارئة أثناء التدريس"). Closing
that gap needs a model that reads the question and the passage
**together**. The reranker in step 5 does this: speed limits on highways now scores 0.006.

### What each choice is worth (ablation, same 73 questions)

| variant | doc #1 | answer/no-answer decision (CV) |
|---|---|---|
| **title prefix + e5 prefixes** (`title_prefix`) | 100% | **91.8%** |
| no document title on chunks (`no_title`) | 98.2% | 87.7% |
| no `query:` / `passage:` prefixes (`no_e5_prefix`) | 98.2% | 80.8% |

## What to change in the production system (Nuxeo + `/v1/query/search`)

- OCR the scanned PDFs before embedding, and check in Elastic that the
  chunk text of each of the 7 documents is real Arabic.
- Expose the **raw cosine** per hit, or apply the threshold **before**
  rank fusion. RRF scores cannot be thresholded.
- Add a reranker stage (`bge-reranker-v2-m3`) on the top-20 hits, and
  decide "no relevant info" on its score, not on the cosine or RRF.
- Replace `min_score: 0.7` with a calibrated value. Start near 0.91 in
  Elastic score (cos 0.82) for e5-large with prefixes, then recalibrate
  with `evaluate.py` for whatever model Nuxeo actually uses.
- Make sure the embedding service adds the `query: ` / `passage: `
  prefixes if the model is e5.
- Scope the experiment to the 7 documents, with a filter or a clean
  index, and refresh the doc IDs.
- Then run `qa_live_api.py` against the API. Its negatives pass only when
  the API returns **no hits**.

## Running it

```bash
pip install pymupdf onnxruntime tokenizers numpy requests scikit-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers sentencepiece
apt-get install tesseract-ocr tesseract-ocr-ara
# model (≈1.3 GB):
mkdir -p models && curl -L https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz | tar xz -C models

python extract.py        # OCR → data/pages.jsonl (committed, so optional)
python build_index.py    # chunks + embeddings (3 variants)
python evaluate.py       # → results_title_prefix.md
python evaluate.py no_e5_prefix   # ablations
python calibration.py            # anisotropy treatments → calibration.md
python rerank.py                 # reranker stage → results_rerank.md (scores cached in data/rerank_scores.json)
SEARCH_API_TOKEN=... python qa_live_api.py   # same test set against the live API
```

`testset.json` holds the questions. Each has `doc` (PDF name, or `null`
for an unanswerable question) and `expected_text` (the evidence passage).
Add more questions there, especially hard negatives, and rerun.
