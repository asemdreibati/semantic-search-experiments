"""Step 4 - fix the "everything scores 0.8" problem at its cause.

e5 embeddings are anisotropic: ~91% of every vector is one shared
direction (norm of the mean vector = 0.91), so any two Arabic texts score
cos ~0.8 and a coffee recipe scores 0.78 against a regulation. This script
compares ways to remove that shared part before scoring, plus a lexical
signal that has a real zero, using the same 5-fold CV protocol as
evaluate.py.

The query-side mean comes from data/background_queries.json (40 generic
questions, not in the test set), the chunk-side mean from the chunks, so
no test label is used to fit any transform.

Usage: python calibration.py   ->  prints a table, writes calibration.md
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import evaluate as ev
from embedder import E5
from textnorm import normalize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------------------------------------------------------------- embeddings
def centered(Q, C, mu_q, mu_c, drop_pcs=0):
    """Subtract each space's mean; optionally also drop the top principal
    directions of the chunks ("all-but-the-top", Mu & Viswanath 2018)."""
    q, c = Q - mu_q, C - mu_c
    if drop_pcs:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        top = vt[:drop_pcs]
        q, c = q - (q @ top.T) @ top, c - (c @ top.T) @ top
    return unit(q) @ unit(c).T


def whitened(Q, C, mu_q, mu_c, bg, k):
    """BERT-whitening (Su et al. 2021): decorrelate and equalise the top-k
    directions, fitted on chunks + background queries."""
    X = np.vstack([C - mu_c, bg - mu_q])
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    W = vt[:k].T / (s[:k] / math.sqrt(len(X)))
    return unit((Q - mu_q) @ W) @ unit((C - mu_c) @ W).T


# ------------------------------------------------------------------- lexical
_STOP = set(normalize(" ".join([
    "في من على إلى عن مع أو ما ماذا هل كيف كيفية متى أين من هي هو التي الذي الذين",
    "هذا هذه ذلك تلك بين بعد قبل عند كل أي لا لم لن إذا ثم حتى وفق خلال طريقة",
])).split())


def tokens(text):
    text = normalize(text)
    text = re.sub("[إأآ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    out = []
    for w in re.findall(r"[ء-ي]+", text):
        w = re.sub(r"^(وال|بال|كال|فال|لل|ال)", "", w) if len(w) > 4 else w
        if len(w) >= 3 and w not in _STOP:
            out.append(w)
    return out


def bm25(queries, docs, k1=1.5, b=0.75):
    docs = [tokens(d) for d in docs]
    df = Counter(w for d in docs for w in set(d))
    avg = sum(map(len, docs)) / len(docs)
    idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    tfs = [Counter(d) for d in docs]
    S = np.zeros((len(queries), len(docs)))
    for i, q in enumerate(queries):
        for j, (tf, d) in enumerate(zip(tfs, docs)):
            S[i, j] = sum(idf[w] * tf[w] * (k1 + 1) /
                          (tf[w] + k1 * (1 - b + b * len(d) / avg))
                          for w in tokens(q) if w in tf)
    return S


# ---------------------------------------------------------------- evaluation
def auc(rows):
    pos = [r["top_score"] for r in rows if r["doc"]]
    neg = [r["top_score"] for r in rows if not r["doc"]]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


def summarise(name, rows, cv):
    pos = [c for c, r in zip(cv, rows) if r["doc"]]
    neg = [c for c, r in zip(cv, rows) if not r["doc"]]
    p = [r["top_score"] for r in rows if r["doc"]]
    n = [r["top_score"] for r in rows if not r["doc"]]
    return {
        "method": name,
        "doc@1": np.mean([r["ranking"][0] == r["doc"] for r in rows if r["doc"]]),
        "AUC": auc(rows),
        "gap": min(p) - max(n),
        "pos": np.mean(pos), "neg": np.mean(neg), "all": np.mean(cv),
    }


def combined_cv(rows_sem, bm_top, k=5, seed=0):
    """Logistic regression on [semantic top score, BM25 top score], with the
    model and its probability cut-off both fitted inside each training fold."""
    import random
    X = np.array([[r["top_score"], b] for r, b in zip(rows_sem, bm_top)])
    y = np.array([r["doc"] is not None for r in rows_sem])
    idx = list(range(len(rows_sem)))
    random.Random(seed).shuffle(idx)
    correct = [None] * len(idx)
    for f in range(k):
        held = idx[f::k]
        train = [i for i in idx if i not in held]
        mu, sd = X[train].mean(0), X[train].std(0)
        lr = LogisticRegression(class_weight="balanced").fit((X[train] - mu) / sd, y[train])
        prob = lr.predict_proba((X - mu) / sd)[:, 1]
        tr_rows = [{**rows_sem[i], "top_score": prob[i]} for i in train]
        T = ev.best_threshold(tr_rows)
        for i in held:
            correct[i] = ev.decide_correct({**rows_sem[i], "top_score": prob[i]}, T)
    return correct


def main():
    chunks = [json.loads(l) for l in (DATA / "chunks.jsonl").open(encoding="utf-8")]
    tests = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    C = np.load(DATA / "emb_title_prefix.npy")
    e5 = E5()
    Q = e5.queries([normalize(t["query"]) for t in tests])
    bg = e5.queries([normalize(q) for q in
                     json.loads((DATA / "background_queries.json").read_text(encoding="utf-8"))])
    mu_q, mu_c = bg.mean(0), C.mean(0)

    sims = {
        "raw cosine (current)": Q @ C.T,
        "centered": centered(Q, C, mu_q, mu_c),
        "centered + drop 1 PC": centered(Q, C, mu_q, mu_c, 1),
        "centered + drop 3 PCs": centered(Q, C, mu_q, mu_c, 3),
        "whitened k=64": whitened(Q, C, mu_q, mu_c, bg, 64),
        "whitened k=128": whitened(Q, C, mu_q, mu_c, bg, 128),
    }
    B = bm25([t["query"] for t in tests], [f"{c['doc']}\n{c['text']}" for c in chunks])
    sims["BM25 only (lexical)"] = B

    results, rows_by = [], {}
    for name, S in sims.items():
        rows = ev.rows_from_sims(tests, S, chunks)
        rows_by[name] = rows
        results.append(summarise(name, rows, ev.cross_validate(rows)))
    for base in ["raw cosine (current)", "centered"]:
        rows = rows_by[base]
        results.append(summarise(f"{base} + BM25 (logistic)", rows,
                                 combined_cv(rows, B.max(1))))

    lines = ["| method | doc #1 | AUC | gap (min pos − max neg) | answerable ok | unanswerable ok | overall (CV) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['method']} | {r['doc@1']:.1%} | {r['AUC']:.3f} | {r['gap']:+.3f} | "
                     f"{r['pos']:.1%} | {r['neg']:.1%} | **{r['all']:.1%}** |")
    print("\n".join(lines))

    best = "centered"
    detail = ["", f"## Scores per question type - `{best}` vs raw", "",
              "| group | raw min / median / max | treated min / median / max |", "|---|---|---|"]
    groups = {"answerable": lambda t: t["doc"],
              "easy negatives": lambda t: t["source"] == "majd (easy negative)",
              "hard negatives": lambda t: t["source"] == "new (hard negative)"}
    for g, f in groups.items():
        a = [r["top_score"] for r in rows_by["raw cosine (current)"] if f(r)]
        b = [r["top_score"] for r in rows_by[best] if f(r)]
        detail.append(f"| {g} | {min(a):.3f} / {np.median(a):.3f} / {max(a):.3f} | "
                      f"{min(b):.3f} / {np.median(b):.3f} / {max(b):.3f} |")
    print("\n".join(detail))
    (HERE / "calibration.md").write_text(
        "# Removing the shared direction (anisotropy) - results\n\n"
        "73 questions, same 5-fold CV protocol as evaluate.py.\n\n"
        + "\n".join(lines + detail) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
