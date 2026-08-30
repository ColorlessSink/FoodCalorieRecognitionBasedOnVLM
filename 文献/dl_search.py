# -*- coding: utf-8 -*-
"""Search arXiv for correct IDs, then download remaining papers."""
import re
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def arxiv_search(q):
    url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) + "&max_results=5"
    req = urllib.request.Request(url, headers=HDRS)
    xml = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    ids = re.findall(r"<id>http://arxiv.org/abs/(.*?)</id>", xml)
    titles = re.findall(r"<title>(.*?)</title>", xml)
    return list(zip(ids, titles[1:]))


for q in ['ti:"A Survey on Food Computing"', 'ti:"FoodSAM"', 'all:"From Pixels to Health"']:
    print("==", q)
    for r in arxiv_search(q):
        print("  ", r)
    time.sleep(2)
