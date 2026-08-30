# -*- coding: utf-8 -*-
"""Round 10: Semantic Scholar paper endpoint for GrabCut direct PDF."""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# GrabCut S2 paper id
url = "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/1015706.1015720?fields=title,openAccessPdf,externalIds"
for attempt in range(8):
    try:
        req = urllib.request.Request(url, headers=HDRS)
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        print(json.dumps(d, indent=2)[:800])
        break
    except Exception as e:
        print(f"retry {attempt+1}: {e}")
        time.sleep(15)
