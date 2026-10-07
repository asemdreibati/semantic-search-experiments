"""Run testset.json through the running demo server, both pipelines, and write
reports in the same layout as the live QA run.

Pass rules as in experiment/qa_report.py: an answerable question passes when
the top document is the expected one; an unanswerable one passes only when
nothing is returned.

Usage: python demo/qa_demo.py   (server running on DEMO_URL, default :8000)
Writes: demo/qa_demo_live_sim.txt, demo/qa_demo_proposed.txt
"""
import json
import os
from pathlib import Path

import requests

from engine import DOC_UUIDS, EXP

HERE = Path(__file__).resolve().parent
URL = os.environ.get("DEMO_URL", "http://127.0.0.1:8000")
LINE, RULE = "-" * 140, "=" * 140
TITLES = {
    "live": "SIMULATED LIVE SYSTEM QA  (bge-m3 + Elasticsearch 8.12 + Nuxeo query, min_score 0.7)",
    "proposed": "PROPOSED SYSTEM QA  (bge-m3 top-10 + bge-reranker-v2-m3, calibrated threshold)",
}


def main():
    tests = json.loads((EXP / "testset.json").read_text(encoding="utf-8"))
    order = [t for t in tests if t["doc"]] + [t for t in tests if not t["doc"]]
    for pipeline in ("live", "proposed"):
        out, passed = [RULE, TITLES[pipeline], RULE], {"pos": 0, "neg": 0}
        for n, t in enumerate(order, 1):
            r = requests.post(f"{URL}/v1/query/search", json={"query": t["query"], "pipeline": pipeline})
            r.raise_for_status()
            hits = r.json()["hits"]
            actual = hits[0]["citation"]["docId"] if hits else None
            expected = DOC_UUIDS.get(t["doc"])
            ok = (actual == expected) if t["doc"] else not hits
            passed["pos" if t["doc"] else "neg"] += ok
            category = t["doc"] or ("easy_negative" if t["source"].startswith("majd") else "hard_negative")
            out += ["", LINE, f"Test Case #{n}",
                    f"Document       : {t['doc']}",
                    f"Category       : {category}",
                    f"Query          : {t['query']}",
                    f"Expected UUID  : {expected}",
                    f"Expected Match : {t['expected_text']}",
                    f"Actual UUID    : {actual}",
                    f"Returned Count : {len(hits)}",
                    f"Status         : {'PASS' if ok else 'FAIL'}"]
            if hits:
                out.append("Top Results:")
                out += [f"  {i}. {h['citation']['docId']} | AICorrespondence - {h['citation']['subject']}"
                        f" | score={h['score']:.4f}" for i, h in enumerate(hits[:5], 1)]
            print(f"{pipeline} #{n} {'PASS' if ok else 'FAIL'}", flush=True)
        total = passed["pos"] + passed["neg"]
        out += ["", RULE, "SUMMARY", RULE,
                f"Total Tests : {len(order)}", f"Passed      : {total}",
                f"Failed      : {len(order) - total}", f"Pass Rate   : {total / len(order):.2%}", "",
                f"Answerable   (#1-57)  : {passed['pos']}/57 right document ranked first",
                f"Unanswerable (#58-73) : {passed['neg']}/16 returned nothing"]
        name = "qa_demo_live_sim.txt" if pipeline == "live" else "qa_demo_proposed.txt"
        (HERE / name).write_text("\n".join(out) + "\n", encoding="utf-8")
        print("\n".join(out[-8:]))


if __name__ == "__main__":
    main()
