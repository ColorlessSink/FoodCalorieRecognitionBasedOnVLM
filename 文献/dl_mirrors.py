# -*- coding: utf-8 -*-
"""Download papers from CVF open access / MDPI / other open mirrors."""
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

PAPERS = [
    # Food-101 (ECCV 2014) - CVF open access
    (
        "https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf",
        "_skip_he.pdf",
    ),
    # (url, filename) pairs below are the real targets
]

TARGETS = [
    # Food-101 ECCV 2014 - try multiple mirrors
    ("https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf",
     "[4] Food-101 - Mining Discriminative Components with Random Forests (ECCV 2014).pdf"),
    # Food Computing survey - ACM CSUR, try author copy
    ("https://arxiv.org/pdf/2112.05646",
     "_probe_survey.pdf"),
    # Recipe1M CVPR 2017 - CVF
    ("https://openaccess.thecvf.com/content_cvpr_2017/papers/Salvador_Learning_Cross-Modal_Embeddings_CVPR_2017_paper.pdf",
     "[25] Recipe1M - Learning Cross-Modal Embeddings for Cooking Recipes and Food Images (CVPR 2017).pdf"),
    # Grad-CAM ICCV 2017 - CVF (already have arXiv)
    # FoodSAM arXiv 2308.05638
    ("https://arxiv.org/pdf/2308.05638",
     "_probe_foodsam.pdf"),
]

HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(url):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    for url, fname in TARGETS:
        try:
            data = fetch(url)
            if data[:5] == b"%PDF-":
                with open(fname, "wb") as f:
                    f.write(data)
                print(f"OK   {len(data)//1024} KB  {fname}  <- {url}")
            else:
                print(f"FAIL (not PDF, {len(data)} B)  {url}")
        except Exception as e:
            print(f"FAIL {e}  {url}")
        time.sleep(2)


if __name__ == "__main__":
    main()
