"""Which embedding model produced the vector Nuxeo sends to Elastic?

The only evidence is the 1024-d query_vector in
"query nuxeo sends to elastic when the user send the prompt.sh"
(unit norm, float16 values). Every embedding model puts all its vectors in
a narrow cone around its own mean direction (anisotropy: for
multilingual-e5-large, cos(vector, mean) ~ 0.85-0.9 for any Arabic text).
Vectors from a *different* model have no reason to lie in that cone.
So for each candidate we embed 40 generic Arabic questions, take their
mean direction, and check whether the Nuxeo vector falls inside the range
the candidate's own vectors occupy.

Usage: python identify_model.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from textnorm import normalize

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

E5_INSTRUCT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
CANDIDATES = {
    # model id: {variant name: encode kwargs}
    "intfloat/multilingual-e5-large": {"query: ": {"prompt": "query: "}, "no prefix": {}},
    "intfloat/multilingual-e5-large-instruct": {"instruct": {"prompt": E5_INSTRUCT}, "no prefix": {}},
    "BAAI/bge-m3": {"no prefix": {}},
    "Snowflake/snowflake-arctic-embed-l-v2.0": {"query: ": {"prompt": "query: "}, "no prefix": {}},
    "Qwen/Qwen3-Embedding-0.6B": {"query prompt": {"prompt_name": "query"}, "no prefix": {}},
}


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def nuxeo_query_vector():
    """The query_vector array from the captured Elastic request."""
    src = (HERE.parent / "query nuxeo sends to elastic when the user send the prompt.sh"
           ).read_text(encoding="utf-8")
    block = src[src.index('"query_vector"'):]
    v = np.array([float(x) for x in block[block.index("[") + 1:block.index("]")].split(",")])
    return v / np.linalg.norm(v)


def main():
    v = nuxeo_query_vector()
    texts = [normalize(q) for q in
             json.loads((DATA / "background_queries.json").read_text(encoding="utf-8"))]
    rows = []
    for model_id, variants in CANDIDATES.items():
        model = SentenceTransformer(model_id, device="cpu")
        for name, kw in variants.items():
            E = unit(model.encode(texts, normalize_embeddings=True, **kw))
            if E.shape[1] != len(v):
                rows.append((model_id, name, E.shape[1], None, None, None))
                continue
            mean = unit(E.mean(0))
            own = E @ mean
            rows.append((model_id, name, E.shape[1], float(own.min()),
                         float(np.median(own)), float(v @ mean)))
            print(rows[-1], flush=True)
        del model

    lines = ["| model | query format | own vectors · mean (min / median) | **Nuxeo vector · mean** | verdict |",
             "|---|---|---|---|---|"]
    for m, n, d, lo, med, nx in rows:
        if nx is None:
            lines.append(f"| {m} | {n} | dim {d} ≠ 1024 | – | excluded |")
            continue
        verdict = "**match**" if nx >= lo - 0.02 else "no"
        lines.append(f"| {m} | {n} | {lo:.3f} / {med:.3f} | **{nx:.3f}** | {verdict} |")
    print("\n".join(lines))
    (HERE / "identify_model.md").write_text(
        "# Identifying the production embedding model\n\n"
        "Nuxeo query vector: 1024-d, unit norm, float16 values.\n\n"
        + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
