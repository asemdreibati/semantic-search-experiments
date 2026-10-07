"""How faithful is the simulation? Run all 73 test questions through the
Nuxeo query on the local Elasticsearch 8.12 and compare with the real
live run (experiment/data/live_run_2026-10-07.json).

Usage: python demo/validate_live_sim.py   (Elasticsearch running, index built)
"""
import json

import numpy as np

from engine import DOC_UUIDS, EXP, Engine


def main():
    tests = json.loads((EXP / "testset.json").read_text(encoding="utf-8"))
    order = [t for t in tests if t["doc"]] + [t for t in tests if not t["doc"]]
    live = json.loads((EXP / "data" / "live_run_2026-10-07.json").read_text(encoding="utf-8"))["cases"]
    eng = Engine(with_reranker=False)
    Q = eng.embed([t["query"] for t in order])
    same_top = same_count = within_one = 0
    sim_pass = {"pos": 0, "neg": 0}
    real_pass = {"pos": 0, "neg": 0}
    for t, real, q in zip(order, live, Q):
        hits = eng.live(t["query"], qvec=q)["hits"]
        top = hits[0]["docId"] if hits else None
        same_top += top == real["top_uuid"]
        same_count += len(hits) == real["returned_count"]
        within_one += abs(len(hits) - real["returned_count"]) <= 1
        key = "pos" if t["doc"] else "neg"
        expected = DOC_UUIDS.get(t["doc"])
        sim_pass[key] += (top == expected) if t["doc"] else (not hits)
        real_pass[key] += (real["top_uuid"] == expected) if t["doc"] else (real["returned_count"] == 0)
    n = len(order)
    print(f"top document same as live : {same_top}/{n}")
    print(f"returned count same       : {same_count}/{n}   (within 1: {within_one}/{n})")
    print(f"answerable  right doc #1  : simulation {sim_pass['pos']}/57   live {real_pass['pos']}/57")
    print(f"unanswerable, nothing back: simulation {sim_pass['neg']}/16   live {real_pass['neg']}/16")


if __name__ == "__main__":
    main()
