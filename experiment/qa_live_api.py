"""Run testset.json against the live search API (POST /v1/query/search).

Fixes three problems in the original QA script:
  1. Expected docs were hard-coded UUIDs that no longer exist in the index
     (none of them appears in any of the 185 hits of the original run), so
     even correct hits were counted as FAIL. Here a hit matches a document
     by DOC_IDS (fill in the current ids) or, failing that, by its subject.
  2. A negative query "passed" whenever the top hit was not one of those
     stale UUIDs - i.e. always. Here a negative passes only if the API
     returns no hits, which is what "no relevant info" means.
  3. The bearer token was committed to git. It is read from the environment.

Usage:  SEARCH_API_TOKEN=... python qa_live_api.py
"""
import json
import os
from pathlib import Path

import requests

from textnorm import normalize

SEARCH_URL = os.environ.get("SEARCH_URL", "http://34.196.113.253:8000/v1/query/search")
TOKEN = os.environ["SEARCH_API_TOKEN"]
TESTSET = Path(__file__).resolve().parent / "testset.json"

# Current docId of each document in the index, keyed by PDF name (without .pdf).
# Leave a value empty to fall back to matching on the hit's subject.
DOC_IDS = {
    "لائحة الاتصالات الرسمية": "220df768-4954-4017-8a75-3bf33d1572c0",
    "سياسة تصنيف البيانات المفتوحة": "",
    "إعادة تشكيل مجلس الوزراء": "",
    "حوكمة التنسيق بين وزارة الصناعة والثروة المعدنية": "",
    "تنظيم الهيئة العامة للطرق لعام 1448هـ": "",
    "السياسة الوطنية لتعزيز السلامة الإسعافية في الأماكن العامة ومقرات العمل لعام 1448هـ": "",
    "معايير التعليم الإلكتروني المحدثة لعام 1448هـ": "",
}


def search(query):
    resp = requests.post(
        SEARCH_URL,
        headers={"Authorization": f"Bearer {TOKEN}", "accept": "application/json"},
        json={"query": query, "topK": 8, "size": 50, "mode": "vector",
              "rerank": False, "expandWindow": 0, "understand": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("hits") or []


def is_doc(hit, doc):
    citation = hit.get("citation", {})
    if DOC_IDS.get(doc):
        return citation.get("docId") == DOC_IDS[doc]
    subject = normalize(citation.get("subject") or "")
    return bool(subject) and (subject in normalize(doc) or normalize(doc) in subject)


def looks_like_rrf(hits):
    """Scores of exactly 1/(60+rank) carry rank only, no similarity."""
    return bool(hits) and all(
        abs(h.get("score", 0) - 1 / (61 + i)) < 1e-9 for i, h in enumerate(hits))


def main():
    tests = json.loads(TESTSET.read_text(encoding="utf-8"))
    results, rrf_seen = [], False
    for t in tests:
        hits = search(t["query"])
        rrf_seen |= looks_like_rrf(hits)
        ok = (not hits) if t["doc"] is None else bool(hits) and is_doc(hits[0], t["doc"])
        results.append((ok, t))
        top = hits[0].get("citation", {}).get("subject") if hits else "-"
        print(f"{'PASS' if ok else 'FAIL'} | {t['doc'] or 'NONE'} | {t['query']} "
              f"| hits={len(hits)} top={top} score={hits[0].get('score') if hits else '-'}")

    pos = [ok for ok, t in results if t["doc"]]
    neg = [ok for ok, t in results if not t["doc"]]
    print(f"\nanswerable   : {sum(pos)}/{len(pos)} correct document ranked #1")
    print(f"unanswerable : {sum(neg)}/{len(neg)} returned no hits")
    if rrf_seen:
        print("\nWARNING: scores are 1/(60+rank) (reciprocal rank fusion). They "
              "carry no similarity information, so no threshold on them can "
              "detect 'no relevant info'. The API must threshold the raw cosine "
              "before fusion (see README.md).")


if __name__ == "__main__":
    main()
