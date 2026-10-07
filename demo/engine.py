"""Two pipelines over the same 7 documents, side by side.

LIVE (simulation of the current Nuxeo system)
    question -> bge-m3 vector -> the exact knn query Nuxeo sends to
    Elasticsearch 8.12 (nested chunk vectors, score_mode max,
    num_candidates 100, min_score 0.7) -> every document above 0.7.
    Nuxeo's ACL / type / trash filters and facet aggregations are left
    out: they do not change the ranking of these 7 documents.

PROPOSED
    question -> bge-m3 -> 10 nearest chunks -> bge-reranker-v2-m3 reads
    (question, chunk) together -> answer from the best chunk if its score
    clears the calibrated threshold, else "no relevant information".

Chunk text comes from experiment/data/chunks.jsonl (built from
experiment/data/pages.jsonl). To mirror the live index more closely,
replace pages.jsonl with the Gemini OCR output and rebuild.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiment"
sys.path.insert(0, str(EXP))

from textnorm import normalize  # noqa: E402

ES_URL = os.environ.get("ES_URL", "http://127.0.0.1:9200")
INDEX = os.environ.get("ES_INDEX", "nuxeo_sim")
EMBED_MODEL = os.environ.get("BGE_M3_DIR", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANKER_DIR", "BAAI/bge-reranker-v2-m3")

LIVE_MIN_SCORE = 0.7          # as in the captured Nuxeo query
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", 10))  # 10 = same accuracy as 20 on the test set, half the time
RERANK_THRESHOLD = 0.0131     # calibrated on testset.json (experiment/results_rerank_bge_m3.md)
NO_ANSWER = "لا توجد معلومات ذات صلة في الوثائق"

# Document ids as they appear in the live index (from the live QA run).
DOC_UUIDS = {
    "لائحة الاتصالات الرسمية": "862116af-472d-453f-9235-e8dc52ae0224",
    "سياسة تصنيف البيانات المفتوحة": "cb850770-7f27-4a47-b6cd-e456e9795734",
    "إعادة تشكيل مجلس الوزراء": "d5245e7f-457b-4a15-844c-b36b8c2c7575",
    "حوكمة التنسيق بين وزارة الصناعة والثروة المعدنية": "d6b2df42-9290-49ad-9be0-19ea6b7b0f72",
    "تنظيم الهيئة العامة للطرق لعام 1448هـ": "891f6473-b661-40d9-915c-e0a531cd7a0a",
    "السياسة الوطنية لتعزيز السلامة الإسعافية في الأماكن العامة ومقرات العمل لعام 1448هـ": "e56b7c5f-6044-4b6c-901b-797aa986bcac",
    "معايير التعليم الإلكتروني المحدثة لعام 1448هـ": "9d250388-4623-4232-8063-f1f5b6f17c92",
}
EMB = "ai_embedding:chunkEmbeddings"

MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {"properties": {
        "ecm:uuid": {"type": "keyword"},
        "dc:title": {"type": "keyword"},
        EMB: {"type": "nested", "properties": {
            "embedding": {"type": "dense_vector", "dims": 1024, "index": True,
                          "similarity": "cosine"},
            "text": {"type": "text", "index": False},
            "page": {"type": "integer"},
            "chunk": {"type": "integer"},
        }},
    }}
}


def live_query(vector, size=10):
    """The Nuxeo request body, filters and aggregations removed."""
    return {
        "from": 0, "size": size,
        "query": {"bool": {"must": [{"nested": {
            "path": EMB, "score_mode": "max",
            "query": {"knn": {"field": f"{EMB}.embedding",
                              "query_vector": [float(x) for x in vector],
                              "num_candidates": 100}},
            "inner_hits": {"size": 5, "_source": {"excludes": [f"{EMB}.embedding"]}},
        }}]}},
        "min_score": LIVE_MIN_SCORE,
        "_source": {"excludes": [f"{EMB}.embedding"]},
    }


class Engine:
    def __init__(self, with_reranker=True):
        from sentence_transformers import SentenceTransformer
        self.chunks = [json.loads(l) for l in (EXP / "data" / "chunks.jsonl").open(encoding="utf-8")]
        self.embedder = SentenceTransformer(EMBED_MODEL, device=self._device())
        self.C = self._chunk_vectors(title=True)   # proposed pipeline: title + chunk
        self.tok = self.reranker = None
        if with_reranker:
            self._load_reranker()

    # ------------------------------------------------------------ set-up
    @staticmethod
    def _device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _chunk_vectors(self, title):
        cache = EXP / "data" / ("emb_bge_m3.npy" if title else "emb_bge_m3_no_title.npy")
        if cache.exists() and len(np.load(cache)) == len(self.chunks):
            return np.load(cache)
        texts = [f"{c['doc']}\n{c['text']}" if title else c["text"] for c in self.chunks]
        vecs = self.embed(texts)
        np.save(cache, vecs)
        return vecs

    def _load_reranker(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL).eval()
        if self._device() == "cuda":
            model = model.half().cuda()
        elif os.environ.get("RERANKER_INT8") == "1":
            # ~10x faster on CPU, but it moves small probabilities by 3-4x
            # (0.004 -> 0.015), right where the 0.0131 threshold sits: on the
            # test set it drops the proposed pipeline from 71/73 to 68/73.
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.reranker = model

    def embed(self, texts):
        return self.embedder.encode([normalize(t) for t in texts], batch_size=8,
                                    normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def build_index(self):
        """One Elasticsearch document per PDF, its chunks as nested vectors
        (chunk text only, no title: the plain form an ingestion pipeline stores)."""
        vecs = self._chunk_vectors(title=False)
        requests.delete(f"{ES_URL}/{INDEX}")
        requests.put(f"{ES_URL}/{INDEX}", json=MAPPING).raise_for_status()
        requests.get(f"{ES_URL}/_cluster/health/{INDEX}",
                     params={"wait_for_status": "green", "timeout": "60s"}).raise_for_status()
        for doc, uuid in DOC_UUIDS.items():
            nested = [{"embedding": vecs[i].tolist(), "text": c["text"], "page": c["page"], "chunk": i}
                      for i, c in enumerate(self.chunks) if c["doc"] == doc]
            body = {"ecm:uuid": uuid, "dc:title": doc, EMB: nested}
            requests.put(f"{ES_URL}/{INDEX}/_doc/{uuid}", json=body).raise_for_status()
        requests.post(f"{ES_URL}/{INDEX}/_refresh").raise_for_status()

    # ---------------------------------------------------------- pipelines
    def live(self, question, qvec=None):
        t0 = time.time()
        qvec = self.embed([question])[0] if qvec is None else qvec
        r = requests.post(f"{ES_URL}/{INDEX}/_search", json=live_query(qvec))
        r.raise_for_status()
        hits = []
        for h in r.json()["hits"]["hits"]:
            inner = h["inner_hits"][EMB]["hits"]["hits"]
            best = inner[0]["_source"] if inner else {}
            hits.append({"docId": h["_source"]["ecm:uuid"], "title": h["_source"]["dc:title"],
                         "score": h["_score"], "page": best.get("page"),
                         "text": best.get("text", "")})
        return {"hits": hits, "ms": int((time.time() - t0) * 1000)}

    def proposed(self, question, qvec=None):
        import torch
        t0 = time.time()
        qvec = self.embed([question])[0] if qvec is None else qvec
        cand = np.argsort(-(self.C @ qvec))[:RERANK_TOP_K]
        q = normalize(question)
        pairs = [[q, f"{self.chunks[j]['doc']}\n{self.chunks[j]['text']}"] for j in cand]
        with torch.inference_mode():
            enc = self.tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(next(self.reranker.parameters()).device) for k, v in enc.items()}
            probs = torch.sigmoid(self.reranker(**enc).logits.view(-1).float()).cpu().numpy()
        order = np.argsort(-probs)
        best = cand[order[0]]
        ranked = [{"title": self.chunks[cand[k]]["doc"], "docId": DOC_UUIDS[self.chunks[cand[k]]["doc"]],
                   "page": self.chunks[cand[k]]["page"], "score": float(probs[k]),
                   "text": self.chunks[cand[k]]["text"]} for k in order[:5]]
        answered = float(probs[order[0]]) >= RERANK_THRESHOLD
        return {"answered": answered,
                "answer": ranked[0] if answered else None,
                "message": None if answered else NO_ANSWER,
                "best_score": float(probs[order[0]]), "threshold": RERANK_THRESHOLD,
                "candidates": ranked, "ms": int((time.time() - t0) * 1000),
                "best_chunk": int(best)}
