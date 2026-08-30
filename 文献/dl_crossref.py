# -*- coding: utf-8 -*-
"""Round 5: use Crossref API (no rate limit issues) to find DOIs and links."""
import json
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "litreview-downloader/1.0 (mailto:student@example.com)"}

QUERIES = [
    ("Food-101 Mining Discriminative Components with Random Forests Bossard", "[4]"),
    ("Saliency Detection A Spectral Residual Approach", "[9]"),
    ("Wide-Slice Residual Networks for Food Recognition Martinel", "[15]"),
    ("Food Detection and Recognition using Deep Convolutional Neural Network Kagaya", "[17]"),
    ("Food Calorie Measurement Using Deep Learning Neural Network Pouladzadeh", "[20]"),
    ("DEEPFOOD Food Image Analysis Dietary Assessment", "[26]"),
    ("GPT-4 Multimodal LLMs Food Energy Estimation Pilot Study", "[32]"),
    ("Kim 2024 Nutrients multimodal LLM food recognition university education", "[34]"),
    ("From Pixels to Health Survey Dietary Assessment Image Analysis", "[38]"),
]


def crossref(q, tag):
    url = (
        "https://api.crossref.org/works?query.bibliographic="
        + urllib.parse.quote(q)
        + "&rows=3&select=title,DOI,URL,container-title,issued"
    )
    try:
        req = urllib.request.Request(url, headers=HDRS)
        data = json.loads(urllib.request.urlopen(req, timeout=60).read())
        print(f"== {tag}")
        for it in data["message"]["items"]:
            t = (it.get("title") or ["?"])[0][:90]
            ct = (it.get("container-title") or ["?"])[0][:40]
            y = it.get("issued", {}).get("date-parts", [["?"]])[0][0]
            print(f"   {t} | {ct} | {y} | doi:{it.get('DOI')}")
    except Exception as e:
        print(f"== {tag} ERROR {e}")


for q, tag in QUERIES:
    crossref(q, tag)
    time.sleep(1)
