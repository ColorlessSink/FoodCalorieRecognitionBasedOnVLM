# -*- coding: utf-8 -*-
"""Round 9: more GrabCut mirrors + spectral residual alternatives."""
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
            with open(fname, "wb") as f:
                f.write(data)
            print(f"OK   {len(data)//1024} KB -> {fname}  ({url})")
            return True
        print(f"NOTPDF {len(data)} B  {url}")
    except Exception as e:
        print(f"ERR {e}  {url}")
    return False


GRABCUT = [
    "https://cseweb.ucsd.edu/classes/wi05/cse291-h/grabcut.pdf",
    "http://www.cs.unc.edu/~lazowski/grabcut.pdf",
    "https://www.cs.princeton.edu/courses/archive/fall14/cos521/papers/grabcut.pdf",
    "https://people.cs.umass.edu/~kalo/papers/labelingPaper/grabcut.pdf",
    "https://www2.cs.sfu.ca/~hamarneh/software/gcut/grabcut.pdf",
    "https://www.robots.ox.ac.uk/~vgg/rg/papers/rother.pdf",
    "http://www.hpl.hp.com/techreports/2004/HPL-2004-99.pdf",
    "https://www.cs.toronto.edu/~jepson/csc2502/papers/rother04grabcut.pdf",
]

done = False
for url in GRABCUT:
    if try_url(url, "[8] GrabCut - Interactive Foreground Extraction using Iterated Graph Cuts (TOG 2004).pdf"):
        done = True
        break
    time.sleep(1)
if not done:
    print("GRABCUT STILL MISSING")
