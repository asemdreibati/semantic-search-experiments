"""Step 2 - chunk pages and embed them.

Each chunk is ~CHUNK_CHARS characters of consecutive paragraphs (never
crossing documents) with one paragraph of overlap, and is prefixed with its
document title so that a chunk like "يعمل بهذا القرار من تاريخه" still
carries what it belongs to.

Output: data/chunks.jsonl, data/emb_<variant>.npy
Usage: python build_index.py [variant ...]   (default: all, see embedder.VARIANTS)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

from embedder import VARIANTS, embedder_for

DATA = Path(__file__).resolve().parent / "data"
CHUNK_CHARS = 800
MIN_CHARS = 40


def paragraphs(text):
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s*\n\s*", " ", p).strip()
        if len(p) >= 3:
            yield p


def chunk_doc(doc, pages):
    units = [(pg["page"], p) for pg in pages for p in paragraphs(pg["text"])]
    chunks, cur = [], []
    for unit in units:
        if cur and sum(len(u[1]) for u in cur) + len(unit[1]) > CHUNK_CHARS:
            chunks.append(cur)
            cur = cur[-1:]  # 1-paragraph overlap
        cur.append(unit)
    if cur:
        chunks.append(cur)
    for c in chunks:
        text = "\n".join(u[1] for u in c)
        if len(text) >= MIN_CHARS:
            yield {"doc": doc, "page": c[0][0], "text": text}


def main():
    pages = [json.loads(l) for l in (DATA / "pages.jsonl").open(encoding="utf-8")]
    by_doc = {}
    for pg in sorted(pages, key=lambda p: (p["doc"], p["page"])):
        by_doc.setdefault(pg["doc"], []).append(pg)
    chunks = [c for doc, pgs in by_doc.items() for c in chunk_doc(doc, pgs)]
    with (DATA / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    for doc in by_doc:
        print(f"  {sum(c['doc'] == doc for c in chunks):4d} chunks  {doc}", file=sys.stderr)

    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        model, title, prefix = embedder_for(name)
        texts = [f"{c['doc']}\n{c['text']}" if title else c["text"] for c in chunks]
        print(f"embedding {len(texts)} chunks [{name}]", file=sys.stderr)
        np.save(DATA / f"emb_{name}.npy", model.passages(texts, prefix=prefix))

if __name__ == "__main__":
    main()
