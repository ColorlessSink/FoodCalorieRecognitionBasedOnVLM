# -*- coding: utf-8 -*-
"""Round 11: CiteSeerX and other aggregators for GrabCut."""
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


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


# CiteSeerX search API
try:
    q = urllib.parse.quote('GrabCut Interactive Foreground Extraction using Iterated Graph Cuts')
    url = f"https://citeseerx.ist.psu.edu/api/search?q={q}&p=0"
    req = urllib.request.Request(url, headers=HDRS)
    import json
    data = json.loads(urllib.request.urlopen(req, timeout=60).read())
    for hit in data.get("results", [])[:5]:
        print("CITESEER:", hit.get("title"))
        print("   ", hit.get("links"))
except Exception as e:
    print("citeseer error:", e)
