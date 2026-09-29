"""Step 5 - second stage: a cross-encoder reads question + chunk TOGETHER.

The bi-encoder (e5) squeezes each chunk into one vector, so it measures
topic similarity. It cannot tell "same topic" from "answers the question",
which is where every remaining error of the cosine threshold sits
(see calibration.md). A cross-encoder attends across both texts at once
and outputs a relevance logit; sigmoid(logit) is a probability-like score
with a meaningful zero.

Pipeline: e5 top-K chunks  ->  bge-reranker-v2-m3 scores each pair  ->
answer if the best reranker score >= T (T by the same 5-fold CV).

Model: BAAI/bge-reranker-v2-m3 (multilingual, XLM-R large), set
RERANKER_DIR or it is fetched from Hugging Face.
Scores are cached in data/rerank_scores.json.

Usage: python rerank.py [variant]   (default title_prefix = e5; bge_m3 = production model)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import evaluate as ev
from embedder import embedder_for
from textnorm import normalize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = DATA / "rerank_scores.json"
MODEL = os.environ.get("RERANKER_DIR", "BAAI/bge-reranker-v2-m3")
TOP_K = 20


def rerank_scores(tests, chunks, Q, C):
    """Return {query: {chunk_index: logit}} for each query's first-stage top-K.
    Cached per (query, chunk) pair, so variants share already-scored pairs."""
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for q, v in cache.items():  # migrate the older {"chunks", "logits"} format
        if "chunks" in v:
            cache[q] = {str(j): l for j, l in zip(v["chunks"], v["logits"])}
    cands = {t["query"]: np.argsort(-(Q[i] @ C.T))[:TOP_K].tolist()
             for i, t in enumerate(tests)}
    todo = [t for t in tests
            if any(str(j) not in cache.get(t["query"], {}) for j in cands[t["query"]])]
    if todo:
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
        for n, t in enumerate(todo, 1):
            done = cache.setdefault(t["query"], {})
            new = [j for j in cands[t["query"]] if str(j) not in done]
            q = normalize(t["query"])
            pairs = [[q, f"{chunks[j]['doc']}\n{chunks[j]['text']}"] for j in new]
            with torch.inference_mode():
                enc = tok(pairs, padding=True, truncation=True, max_length=512,
                          return_tensors="pt")
                logits = model(**enc).logits.view(-1).float().tolist()
            done.update({str(j): l for j, l in zip(new, logits)})
            print(f"  reranked {n}/{len(todo)} ({len(new)} new pairs)", flush=True)
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {q: {j: cache[q][str(j)] for j in c} for q, c in cands.items()}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "title_prefix"
    chunks = [json.loads(l) for l in (DATA / "chunks.jsonl").open(encoding="utf-8")]
    tests = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    C = np.load(DATA / f"emb_{variant}.npy")
    q_cache = DATA / f"q_{variant}.npy"
    if q_cache.exists() and len(np.load(q_cache)) == len(tests):
        Q = np.load(q_cache)
    else:
        model, _, prefix = embedder_for(variant)
        Q = model.queries([normalize(t["query"]) for t in tests], prefix=prefix)
        np.save(q_cache, Q)
    scores = rerank_scores(tests, chunks, Q, C)

    # chunks outside the first-stage top-K get -1, below any sigmoid score
    S = np.full((len(tests), len(chunks)), -1.0)
    for i, t in enumerate(tests):
        idx = list(scores[t["query"]])
        S[i, idx] = 1 / (1 + np.exp(-np.array([scores[t["query"]][j] for j in idx])))
    rows = ev.rows_from_sims(tests, S, chunks)
    raw = ev.score_queries(variant)

    out = [f"# Results - `{variant}` top-20 + bge-reranker-v2-m3\n",
           "| | cosine only | + reranker |", "|---|---|---|"]
    for label, f in [("correct document ranked #1",
                      lambda rs: np.mean([r["ranking"][0] == r["doc"] for r in rs if r["doc"]])),
                     ("evidence text in top-5 chunks",
                      lambda rs: np.mean([r["evidence_top5"] for r in rs if r["doc"]]))]:
        out.append(f"| {label} | {f(raw):.1%} | {f(rows):.1%} |")
    for name, rs in [("raw", raw), ("rr", rows)]:
        cv = ev.cross_validate(rs)
        pos = np.mean([c for c, r in zip(cv, rs) if r["doc"]])
        neg = np.mean([c for c, r in zip(cv, rs) if not r["doc"]])
        if name == "raw":
            base = (pos, neg, np.mean(cv))
        else:
            out += [f"| answerable -> answered from right doc (CV) | {base[0]:.1%} | {pos:.1%} |",
                    f"| unanswerable -> no info (CV) | {base[1]:.1%} | {neg:.1%} |",
                    f"| **overall (CV)** | **{base[2]:.1%}** | **{np.mean(cv):.1%}** |"]
    T = ev.best_threshold(rows)
    out += ["", f"Reranker threshold fitted on all queries: **{T:.4f}**", "",
            "## Best reranker score per question type", "",
            "| group | min | median | max |", "|---|---|---|---|"]
    groups = {"answerable": lambda r: r["doc"],
              "easy negatives": lambda r: r["source"] == "majd (easy negative)",
              "hard negatives": lambda r: r["source"] == "new (hard negative)"}
    for g, f in groups.items():
        v = [r["top_score"] for r in rows if f(r)]
        out.append(f"| {g} | {min(v):.3f} | {np.median(v):.3f} | {max(v):.3f} |")
    out += ["", "## Per query", "",
            "| ok | expected doc | query | cosine | reranker | top doc |", "|---|---|---|---|---|---|"]
    for r, b in sorted(zip(rows, raw), key=lambda x: (x[0]["doc"] is None, -x[0]["top_score"])):
        ok = "✅" if ev.decide_correct(r, T) else "❌"
        top = r["ranking"][0] if r["top_score"] >= T else f"_({ev.NO_ANSWER})_"
        out.append(f"| {ok} | {r['doc'] or 'NONE'} | {r['query']} | {b['top_score']:.3f} | "
                   f"{r['top_score']:.3f} | {top} |")
    (HERE / ("results_rerank.md" if variant == "title_prefix" else f"results_rerank_{variant}.md")).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:24]))


if __name__ == "__main__":
    main()
