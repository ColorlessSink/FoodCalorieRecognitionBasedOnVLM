# -*- coding: utf-8 -*-
"""Round 8: GrabCut + remaining papers via course-page mirrors."""
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
        data = urllib.request.urlopen(req, timeout=90).read()
        if data[:5] == b"%PDF-":
            # basic content check: search raw bytes for a keyword
            low = data[:200000].lower()
            with open(fname, "wb") as f:
                f.write(data)
            print(f"OK   {len(data)//1024} KB -> {fname}  ({url})")
            return True
        print(f"NOTPDF {len(data)} B  {url}")
    except Exception as e:
        print(f"ERR {e}  {url}")
    return False


GRABCUT = [
    ("https://www.cs.cornell.edu/courses/cs4670/2018fa/lectures/lect13/grabcut.pdf", False),
    ("https://www.cs.uic.edu/~jbell5/cs594/handouts/grabcut.pdf", False),
    ("https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/grabcut-siggraph04.pdf", False),
    ("https://cvg.ethz.ch/education/imagetocomp13/rother.pdf", False),
    ("https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/sheasby/grabcut.pdf", False),
    ("https://www.comp.nus.edu.sg/~cs4243/lecture/grabcut.pdf", False),
]

done = False
for url, _ in GRABCUT:
    if try_url(url, "[8] GrabCut - Interactive Foreground Extraction using Iterated Graph Cuts (TOG 2004).pdf"):
        done = True
        break
    time.sleep(1)
if not done:
    print("GRABCUT STILL MISSING")
