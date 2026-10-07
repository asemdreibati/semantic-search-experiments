"""Write a QA report in the same layout as the live-system QA run, from the
local pipeline, so the two can be compared case by case.

Numbering follows the live report: answerable questions in testset.json
order (#1-57), then the unanswerable ones (#58-73).

Pass rules:
  answerable   -> PASS when the top returned document is the expected one
  unanswerable -> PASS only when NOTHING is returned (Returned Count 0).
                  Returning hits for a question the documents cannot answer
                  is exactly the failure "no relevant info" is meant to catch.

Each question is judged with a threshold fitted on the OTHER questions
(same 5-fold split as evaluate.py), so the pass rate is an honest estimate
for unseen questions, not a threshold tuned on the answers.

Usage: python qa_report.py [rerank|cosine]   (default rerank; variant bge_m3)
Writes: qa_report_bge_m3_<mode>.txt
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

import evaluate as ev

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VARIANT = "bge_m3"
LINE = "-" * 140
RULE = "=" * 140

# Document ids in the live index (from the live QA report).
DOC_UUIDS = {
    "لائحة الاتصالات الرسمية": "862116af-472d-453f-9235-e8dc52ae0224",
    "سياسة تصنيف البيانات المفتوحة": "cb850770-7f27-4a47-b6cd-e456e9795734",
    "إعادة تشكيل مجلس الوزراء": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
    "حوكمة التنسيق بين وزارة الصناعة والثروة المعدنية": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
    "تنظيم الهيئة العامة للطرق لعام 1448هـ": "891f6473-b661-40d9-915c-e0a531cd7a0a",
    "السياسة الوطنية لتعزيز السلامة الإسعافية في الأماكن العامة ومقرات العمل لعام 1448هـ": "e56b7c5f-6044-4b6c-901b-797aa986bcac",
    "معايير التعليم الإلكتروني المحدثة لعام 1448هـ": "9d250388-4623-4232-8063-f1f5b6f17c92",
}
CATEGORY = {
    "تنظيم الهيئة العامة للطرق لعام 1448هـ": "تنظيم الهيئة العامة للطرق",
    "السياسة الوطنية لتعزيز السلامة الإسعافية في الأماكن العامة ومقرات العمل لعام 1448هـ": "السياسة الوطنية لتعزيز السلامة الإسعافية",
    "معايير التعليم الإلكتروني المحدثة لعام 1448هـ": "معايير التعليم الإلكتروني",
}


def score_matrix(mode, tests, chunks):
    if mode == "cosine":
        return np.load(DATA / f"q_{VARIANT}.npy") @ np.load(DATA / f"emb_{VARIANT}.npy").T
    # reranker probability on the bge-m3 top-20 chunks, -1 elsewhere
    import rerank
    Q = np.load(DATA / f"q_{VARIANT}.npy")
    scores = rerank.rerank_scores(tests, chunks, Q, np.load(DATA / f"emb_{VARIANT}.npy"))
    S = np.full((len(tests), len(chunks)), -1.0)
    for i, t in enumerate(tests):
        idx = list(scores[t["query"]])
        S[i, idx] = 1 / (1 + np.exp(-np.array([scores[t["query"]][j] for j in idx])))
    return S


def cv_thresholds(rows, k=5, seed=0):
    """Threshold each question is judged with: fitted on the other folds."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    T = [None] * len(rows)
    for f in range(k):
        held = set(idx[f::k])
        t = ev.best_threshold([rows[i] for i in idx if i not in held])
        for i in held:
            T[i] = t
    return T


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rerank"
    tests = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in (DATA / "chunks.jsonl").open(encoding="utf-8")]
    S = score_matrix(mode, tests, chunks)
    rows = ev.rows_from_sims(tests, S, chunks)
    T = cv_thresholds(rows)

    docs = sorted({c["doc"] for c in chunks})
    doc_idx = np.array([docs.index(c["doc"]) for c in chunks])
    order = [i for i, t in enumerate(tests) if t["doc"]] + \
            [i for i, t in enumerate(tests) if not t["doc"]]

    label = {"rerank": "bge-m3 top-20 + bge-reranker-v2-m3", "cosine": "bge-m3 cosine only"}[mode]
    out = [RULE, f"LOCAL SEMANTIC SEARCH QA  ({label})", RULE]
    passed = {"pos": 0, "neg": 0}
    for n, i in enumerate(order, 1):
        t = tests[i]
        per_doc = np.full(len(docs), -1.0)
        np.maximum.at(per_doc, doc_idx, S[i])
        hits = [(docs[j], per_doc[j]) for j in np.argsort(-per_doc) if per_doc[j] >= T[i]]
        actual = DOC_UUIDS[hits[0][0]] if hits else None
        expected = DOC_UUIDS.get(t["doc"])
        ok = (not hits) if t["doc"] is None else actual == expected
        passed["neg" if t["doc"] is None else "pos"] += ok
        if t["doc"]:
            category = CATEGORY.get(t["doc"], t["doc"])
        else:
            category = "easy_negative" if t["source"].startswith("majd") else "hard_negative"
        out += ["", LINE, f"Test Case #{n}",
                f"Document       : {t['doc']}",
                f"Category       : {category}",
                f"Query          : {t['query']}",
                f"Expected UUID  : {expected}",
                f"Expected Match : {t['expected_text']}",
                f"Actual UUID    : {actual}",
                f"Returned Count : {len(hits)}",
                f"Best Score     : {rows[i]['top_score']:.4f}   (threshold {T[i]:.4f})",
                f"Status         : {'PASS' if ok else 'FAIL'}"]
        if hits:
            out.append("Top Results:")
            out += [f"  {r}. {DOC_UUIDS[d]} | AICorrespondence - {d} | score={s:.4f}"
                    for r, (d, s) in enumerate(hits[:5], 1)]

    n_pos = sum(1 for t in tests if t["doc"])
    n_neg = len(tests) - n_pos
    total = passed["pos"] + passed["neg"]
    out += ["", RULE, "SUMMARY", RULE,
            f"Total Tests : {len(tests)}",
            f"Passed      : {total}",
            f"Failed      : {len(tests) - total}",
            f"Pass Rate   : {total / len(tests):.2%}",
            "",
            f"Answerable   (#1-{n_pos})  : {passed['pos']}/{n_pos} right document ranked first",
            f"Unanswerable (#{n_pos + 1}-{len(tests)}) : {passed['neg']}/{n_neg} returned nothing",
            f"Thresholds  : {min(T):.4f} - {max(T):.4f}, each fitted on the other 4 of 5 folds"]
    path = HERE / f"qa_report_{VARIANT}_{mode}.txt"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[-12:]))


if __name__ == "__main__":
    main()
