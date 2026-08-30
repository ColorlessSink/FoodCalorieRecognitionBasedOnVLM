# -*- coding: utf-8 -*-
"""Round 4: BMVC archive, MDPI with browser headers, other mirrors."""
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

HDRS_BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

TARGETS = [
    # Kagaya BMVC 2015 - BMVC archive (open). paper number unknown, try index first
    # Pouladzadeh TIM 2014 - arXiv? No. Try uOttawa author copy
    (
        "https://www.researchgate.net/publication/262395558",
        "_skip.pdf",
    ),
]


def fetch(url, fname, hdrs=HDRS_BROWSER):
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if data[:5] == b"%PDF-":
        with open(fname, "wb") as f:
            f.write(data)
        return f"OK {len(data)//1024} KB -> {fname}"
    return f"NOTPDF ({len(data)} B) {url}"


# BMVC 2015 archive: find Kagaya paper
try:
    req = urllib.request.Request(
        "https://www.bmva-archive.org/bmvc/2015/papers/index.html", headers=HDRS_BROWSER
    )
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    # find lines mentioning Kagaya or food
    for line in html.splitlines():
        if "agaya" in line or "ood" in line.lower() and "etect" in line:
            print("BMVC2015:", line.strip()[:200])
except Exception as e:
    print("bmvc2015 index error:", e)
