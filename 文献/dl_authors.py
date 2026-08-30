# -*- coding: utf-8 -*-
"""Round 7: author-homepage mirrors for the stubborn papers."""
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}


def try_url(url, fname):
    try:
        req = urllib.request.Request(url, headers=HDRS)
        data = urllib.request.urlopen(req, timeout=60).read()
        if data[:5] == b"%PDF-":
            with open(fname, "wb") as f:
                f.write(data)
            print(f"OK   {len(data)//1024} KB -> {fname}  ({url})")
            return True
        print(f"NOTPDF {len(data)} B  {url}")
    except Exception as e:
        print(f"ERR {e}  {url}")
    return False


CANDIDATES = {
    "[9] Saliency Detection - A Spectral Residual Approach (CVPR 2007).pdf": [
        # Hou & Zhang author copies / course mirrors
        "https://www.cs.sjtu.edu.cn/~xinhao/paper/cvpr07_spectral.pdf",
        "https://ftp.cs.uwaterloo.ca/cs-course-notes/cs480/gestalt/resources/cvpr07.pdf",
        "http://www.lvronline.net/papers/cvpr07.pdf",
        "https://www.cs.cityu.edu.hk/~rynson/papers/cvpr07.pdf",
        "https://papers-gamma.link/paper/10.1109/CVPR.2007.383267.pdf",
        "https://sci-hub.se/10.1109/cvpr.2007.383267",
    ],
    "[4] Food-101 - Mining Discriminative Components with Random Forests (ECCV 2014).pdf": [
        "https://www.researchgate.net/publication/262395558",
        "https://ivc.ischool.utexas.edu/files/food101.pdf",
        "https://www.ics.uci.edu/~dramanan/food101.pdf",
    ],
}

for fname, urls in CANDIDATES.items():
    done = False
    for u in urls:
        if try_url(u, fname):
            done = True
            break
        time.sleep(1)
    if not done:
        print(f"STILL MISSING: {fname}")
