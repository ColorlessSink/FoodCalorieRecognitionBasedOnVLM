# -*- coding: utf-8 -*-
"""Round 3: download remaining papers with corrected arXiv IDs."""
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TARGETS = [
    # Food Computing survey (ACM CSUR 2019) - arXiv version
    ("1808.07202", "[13] A Survey on Food Computing (ACM Computing Surveys 2019).pdf"),
    # FoodSAM (not FoodSeg103 itself, but the SAM-for-food followup we cite alongside)
    ("2308.05938", "_probe_foodsam2.pdf"),
    # Food-101 via CVF open access (ECCV 2014 is on cv-foundation)
    # MiLaS v1 "Vision Transformers for Single-Image Depth Estimation" not needed; have v2
    # GPT-4V food eval: Attipa Nutrients 2024 - MDPI open access
    (
        "https://www.mdpi.com/2072-6643/16/14/2248/pdf",
        "_probe_attipa.pdf",
    ),
]


def fetch_pdf(url):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    for url_or_id, fname in TARGETS:
        url = url_or_id if url_or_id.startswith("http") else f"https://arxiv.org/pdf/{url_or_id}"
        try:
            data = fetch_pdf(url)
            if data[:5] == b"%PDF-":
                with open(fname, "wb") as f:
                    f.write(data)
                print(f"OK   {len(data)//1024} KB  {fname}")
            else:
                print(f"FAIL not-PDF ({len(data)} B)  {url}")
        except Exception as e:
            print(f"FAIL {e}  {url}")
        time.sleep(2)


if __name__ == "__main__":
    main()
