# -*- coding: utf-8 -*-
"""Retry Semantic Scholar with longer waits between calls (429 rate limit)."""
import json
import sys
import time
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

QUERIES = [
    ("Food-101 Mining Discriminative Components with Random Forests", "[4] Food-101"),
    ("Saliency Detection A Spectral Residual Approach Hou Zhang", "[9] Spectral Residual"),
    ("Wide-Slice Residual Networks for Food Recognition", "[15] WiSE-ResNet"),
    ("Food Detection and Recognition Using Convolutional Neural Networks Kagaya", "[17] Kagaya BMVC15"),
    ("Food Calorie Measurement Using Deep Learning Neural Network Pouladzadeh", "[20] Pouladzadeh"),
    ("DeepFood Food Image Analysis and Dietary Assessment", "[26] DeepFood"),
    ("GPT-4 and Multimodal LLMs in Food Energy Estimation Pilot Study Attipa", "[32] Attipa"),
    ("Nutrients multimodal large language models food recognition education Kim", "[34] Kim MDPI"),
    ("From Pixels to Health Dietary Assessment Image Analysis Survey", "[38] Zhang survey"),
]


def search(q, tag):
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search?query="
        + urllib.parse.quote(q)
        + "&fields=title,year,openAccessPdf&limit=3"
    )
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            data = json.loads(urllib.request.urlopen(req, timeout=60).read())
            print(f"== {tag}")
            for p in data.get("data", []):
                oa = p.get("openAccessPdf")
                print(f"   {p.get('title')} ({p.get('year')})")
                print(f"      pdf: {oa.get('url') if oa else 'NONE'}")
            return
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"== {tag}  retry {attempt+1} after {wait}s ({e})")
            time.sleep(wait)


for q, tag in QUERIES:
    search(q, tag)
    time.sleep(15)
