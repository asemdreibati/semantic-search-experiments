"""Build demo/static/demo.html: a single page with the precomputed results of
all 73 test questions, which opens in any browser with nothing installed.

    python demo/static/build_static.py              # needs the demo running on :8000
    python demo/static/build_static.py data.json    # rebuild from saved results

Writes demo/static/demo.html and demo/static/demo_data.json.
"""
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from engine import DOC_UUIDS, EXP  # noqa: E402

URL = "http://127.0.0.1:8000/api/compare"


def cut(s, n):
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


def fetch():
    tests = json.loads((EXP / "testset.json").read_text(encoding="utf-8"))
    order = [x for x in tests if x["doc"]] + [x for x in tests if not x["doc"]]
    live = json.loads((EXP / "data" / "live_run_2026-10-07.json").read_text(encoding="utf-8"))["cases"]
    title = {v: k for k, v in DOC_UUIDS.items()}
    out = []
    for n, (x, lv) in enumerate(zip(order, live), 1):
        d = requests.post(URL, json={"query": x["query"]}).json()
        p = d["proposed"]
        out.append({
            "n": n, "q": x["query"], "doc": x["doc"], "exp": x["expected_text"],
            "kind": "ans" if x["doc"] else ("easy" if x["source"].startswith("majd") else "hard"),
            "real": {"count": lv["returned_count"], "top": title.get(lv["top_uuid"])},
            "sim": [{"t": h["title"], "s": round(h["score"], 3), "p": h["page"], "x": cut(h["text"], 260)}
                    for h in d["live"]["hits"]],
            "pro": {"ok": p["answered"], "best": round(p["best_score"], 4),
                    "a": ({"t": p["answer"]["title"], "p": p["answer"]["page"],
                           "s": round(p["answer"]["score"], 4), "x": cut(p["answer"]["text"], 1100)}
                          if p["answered"] else None),
                    "c": [{"t": c["title"], "p": c["page"], "s": round(c["score"], 4), "x": cut(c["text"], 180)}
                          for c in p["candidates"][:5]]},
        })
        print(n, flush=True)
    return out


def main():
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")) if len(sys.argv) > 1 else fetch()
    (HERE / "demo_data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    page = (HERE / "template.html").read_text(encoding="utf-8")
    page = page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    standalone = ('<!doctype html>\n<html lang="ar" dir="rtl">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
                  + page + "\n</body>\n</html>\n")
    (HERE / "demo.html").write_text(standalone, encoding="utf-8")
    print(f"wrote {HERE / 'demo.html'} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
