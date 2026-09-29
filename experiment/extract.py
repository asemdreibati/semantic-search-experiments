"""Step 1 - PDF -> clean Arabic text, one record per page.

5 of the 7 PDFs are scanned images (no text layer at all) and the other 2
have broken Arabic text layers (presentation-form glyphs, reversed lam-alef),
so every page is OCR'd with Tesseract (ara) for uniform quality. ("ara+eng"
tested worse: Arabic words get misread as Latin garbage.)

Output: data/pages.jsonl  {"doc", "page", "text"}
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from textnorm import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data" / "pages.jsonl"
DPI = 300


def ocr_page(args):
    pdf, page_no = args
    page = pymupdf.open(pdf)[page_no]
    png = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY).tobytes("png")
    text = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "ara", "--psm", "3"],
        input=png, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    return {"doc": Path(pdf).stem, "page": page_no + 1, "text": normalize(text)}


def main():
    pdfs = sorted(ROOT.glob("*.pdf"))
    jobs = [(str(p), i) for p in pdfs for i in range(len(pymupdf.open(p)))]
    print(f"OCR {len(jobs)} pages from {len(pdfs)} PDFs", file=sys.stderr)
    # one Tesseract thread per worker process, else OpenMP oversubscribes the CPU
    os.environ["OMP_THREAD_LIMIT"] = "1"
    recs = []
    with ProcessPoolExecutor() as pool:
        for n, rec in enumerate(pool.map(ocr_page, jobs), 1):
            recs.append(rec)
            print(f"  {n}/{len(jobs)} {rec['doc']} p{rec['page']}", file=sys.stderr, flush=True)
    with OUT.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
