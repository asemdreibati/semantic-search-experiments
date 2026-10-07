"""Demo server: the current system (simulated) next to the proposed one.

    python demo/server.py            # then open http://127.0.0.1:8000

Endpoints
    GET  /                    the comparison page
    GET  /api/examples        the 73 test questions with their expected answer
    POST /api/compare         {"query": "..."} -> {"live": ..., "proposed": ...}
    POST /v1/query/search     live-API-shaped, so the existing QA scripts can
                              point at it: {"query": "...", "pipeline": "live"|"proposed"}
                              -> {"hits": [{"score", "citation": {"docId", "subject"}, "text"}], "total"}
"""
import json
import os
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine import DOC_UUIDS, ES_URL, EXP, INDEX, Engine

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Semantic search demo")
engine: Engine = None


class Ask(BaseModel):
    query: str
    pipeline: str = "live"


@app.on_event("startup")
def load():
    global engine
    engine = Engine()
    if requests.get(f"{ES_URL}/{INDEX}").status_code == 404:
        engine.build_index()


@app.get("/")
def page():
    return FileResponse(HERE / "index.html")


@app.get("/api/examples")
def examples():
    tests = json.loads((EXP / "testset.json").read_text(encoding="utf-8"))
    return [{"query": t["query"], "doc": t["doc"], "expected": t["expected_text"],
             "kind": "answerable" if t["doc"] else
                     ("easy_negative" if t["source"].startswith("majd") else "hard_negative")}
            for t in tests]


@app.post("/api/compare")
def compare(ask: Ask):
    qvec = engine.embed([ask.query])[0]
    return {"query": ask.query,
            "live": engine.live(ask.query, qvec=qvec),
            "proposed": engine.proposed(ask.query, qvec=qvec)}


@app.post("/v1/query/search")
def search(ask: Ask):
    if ask.pipeline == "proposed":
        r = engine.proposed(ask.query)
        hits = [r["answer"]] if r["answered"] else []
    else:
        hits = engine.live(ask.query)["hits"]
    return {"total": len(hits),
            "hits": [{"score": h["score"], "text": h["text"], "page": h["page"],
                      "citation": {"docId": h["docId"], "subject": h["title"]}} for h in hits]}


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
