# -*- coding: utf-8 -*-
"""Round 6: DOI-based open-access attempts via unpaywall + direct."""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# (doi, filename)
DOIS = [
    ("10.1007/978-3-319-10599-4_29", "[4] Food-101 - Mining Discriminative Components with Random Forests (ECCV 2014).pdf"),
    ("10.1109/cvpr.2007.383267", "[9] Saliency Detection - A Spectral Residual Approach (CVPR 2007).pdf"),
    ("10.1109/wacv.2018.00068", "[15] WiSE-ResNet - Wide-Slice Residual Networks for Food Recognition (WACV 2018).pdf"),
    ("10.1145/2647868.2654970", "[17] Kagaya - Food Detection and Recognition Using CNN (2014).pdf"),
    ("10.1109/i2mtc.2016.7520547", "[20] Pouladzadeh - Food Calorie Measurement Using Deep Learning Neural Network (I2MTC 2016).pdf"),
    ("10.1109/access.2020.2973625", "[26] DeepFood - Food Image Analysis and Dietary Assessment (IEEE Access 2020).pdf"),
]


def unpaywall(doi):
    url = f"https://api.unpaywall.org/v2/{doi}?email=student@example.com"
    req = urllib.request.Request(url, headers=HDRS)
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        loc = d.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception as e:
        return None


for doi, fname in DOIS:
    loc = unpaywall(doi)
    print(f"{doi} -> {loc}")
    if loc:
        try:
            req = urllib.request.Request(loc, headers=HDRS)
            data = urllib.request.urlopen(req, timeout=120).read()
            if data[:5] == b"%PDF-":
                with open(fname, "wb") as f:
                    f.write(data)
                print(f"   OK {len(data)//1024} KB -> {fname}")
            else:
                print(f"   NOTPDF {len(data)} B")
        except Exception as e:
            print(f"   fetch error: {e}")
    time.sleep(1)
