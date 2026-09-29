"""Step 3 - evaluate retrieval AND the "no relevant info" decision.

For every query in testset.json:
  * rank chunks by cosine, collapse to documents (max chunk score per doc)
  * positives: is the expected document ranked 1st / in top 3? does one of
    the top 5 chunks actually contain the expected evidence text?
  * decision: answer only if best chunk cosine >= T, else "لا توجد معلومات"

T is not guessed: it is chosen by 5-fold cross-validation (pick T on 4/5 of
the queries, measure on the held-out 1/5), so the reported accuracy is an
honest estimate for unseen questions.

Usage: python evaluate.py [variant]   (default: title_prefix)
Writes: results_<variant>.md
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from embedder import E5
from textnorm import normalize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NO_ANSWER = "لا توجد معلومات ذات صلة في الوثائق"


def ngrams(s, n=3):
    s = "".join(s.split())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def evidence_found(expected, chunk_texts, min_overlap=0.6):
    """OCR text is noisy, so match on character-trigram overlap, not equality."""
    want = ngrams(normalize(expected))
    return any(len(want & ngrams(t)) / max(len(want), 1) >= min_overlap
               for t in chunk_texts)


def score_queries(variant):
    chunks = [json.loads(l) for l in (DATA / "chunks.jsonl").open(encoding="utf-8")]
    emb = np.load(DATA / f"emb_{variant}.npy")
    tests = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    q = E5().queries([normalize(t["query"]) for t in tests],
                     prefix=variant != "no_e5_prefix")
    return rows_from_sims(tests, q @ emb.T, chunks)


def rows_from_sims(tests, sims, chunks):
    """Rank chunks per query by `sims` (queries x chunks), collapse to docs."""
    docs = sorted({c["doc"] for c in chunks})
    doc_idx = np.array([docs.index(c["doc"]) for c in chunks])
    rows = []
    for t, s in zip(tests, sims):
        per_doc = np.full(len(docs), -1.0)
        np.maximum.at(per_doc, doc_idx, s)
        ranking = [docs[i] for i in np.argsort(-per_doc)]
        top5 = np.argsort(-s)[:5]
        rows.append({
            **t,
            "top_score": float(s.max()),
            "ranking": ranking,
            "doc_scores": {docs[i]: float(per_doc[i]) for i in np.argsort(-per_doc)[:3]},
            "top_chunk": chunks[top5[0]]["text"][:160].replace("\n", " "),
            "evidence_top5": bool(t.get("expected_text")) and evidence_found(
                t["expected_text"], [chunks[i]["text"] for i in top5]),
        })
    return rows


def decide_correct(r, T):
    answered = r["top_score"] >= T
    if r["doc"] is None:
        return not answered
    return answered and r["ranking"][0] == r["doc"]


def best_threshold(rows):
    """Maximise balanced accuracy (positives and negatives weigh the same)."""
    pos = [r for r in rows if r["doc"]]
    neg = [r for r in rows if not r["doc"]]
    cands = sorted({round(r["top_score"], 4) for r in rows})
    def bal(T):
        return (np.mean([decide_correct(r, T) for r in pos])
                + np.mean([decide_correct(r, T) for r in neg])) / 2
    return max(cands, key=bal)


def cross_validate(rows, k=5, seed=0):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    correct = [None] * len(rows)
    for f in range(k):
        held = set(idx[f::k])
        T = best_threshold([rows[i] for i in idx if i not in held])
        for i in held:
            correct[i] = decide_correct(rows[i], T)
    return correct


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "title_prefix"
    rows = score_queries(variant)
    pos = [r for r in rows if r["doc"]]
    neg = [r for r in rows if not r["doc"]]
    T = best_threshold(rows)
    cv = cross_validate(rows)
    cv_pos = np.mean([c for c, r in zip(cv, rows) if r["doc"]])
    cv_neg = np.mean([c for c, r in zip(cv, rows) if not r["doc"]])

    out = [f"# Results - variant `{variant}`\n",
           f"{len(pos)} answerable queries, {len(neg)} unanswerable queries, "
           f"{len({r['doc'] for r in pos})} documents.\n",
           "## Retrieval (answerable queries)\n",
           "| metric | value |", "|---|---|",
           f"| correct document ranked #1 | {np.mean([r['ranking'][0] == r['doc'] for r in pos]):.1%} |",
           f"| correct document in top 3 | {np.mean([r['doc'] in r['ranking'][:3] for r in pos]):.1%} |",
           f"| expected evidence text in top-5 chunks | {np.mean([r['evidence_top5'] for r in pos]):.1%} |",
           "\n## \"No relevant info\" decision\n",
           f"Best-chunk cosine: answerable min/median = "
           f"{min(r['top_score'] for r in pos):.3f} / {np.median([r['top_score'] for r in pos]):.3f}; "
           f"unanswerable median/max = {np.median([r['top_score'] for r in neg]):.3f} / "
           f"{max(r['top_score'] for r in neg):.3f}\n",
           f"Threshold fitted on all queries: **T = {T:.4f}**\n",
           "| (5-fold cross-validated) | value |", "|---|---|",
           f"| answerable -> answered from the right document | {cv_pos:.1%} |",
           f"| unanswerable -> \"{NO_ANSWER}\" | {cv_neg:.1%} |",
           f"| overall | {np.mean(cv):.1%} |",
           "\n## Per query (threshold T)\n",
           "| ok | expected doc | query | best cosine | top doc | evidence@5 |",
           "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["doc"] is None, r["doc"] or "", -r["top_score"])):
        ok = "✅" if decide_correct(r, T) else "❌"
        top = r["ranking"][0] if r["top_score"] >= T else f"_({NO_ANSWER})_"
        ev = "" if not r["doc"] else ("yes" if r["evidence_top5"] else "no")
        out.append(f"| {ok} | {r['doc'] or 'NONE'} | {r['query']} | {r['top_score']:.3f} | {top} | {ev} |")
    (HERE / f"results_{variant}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:22]))


if __name__ == "__main__":
    main()
