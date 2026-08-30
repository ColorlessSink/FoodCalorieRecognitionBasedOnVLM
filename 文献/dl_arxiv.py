# -*- coding: utf-8 -*-
"""Batch download papers from arXiv into the current directory."""
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

PAPERS = [
    # (arxiv_id, filename)
    ("2304.08485", "[10] LLaVA - Visual Instruction Tuning (NeurIPS 2023).pdf"),
    ("2301.12597", "[11] BLIP-2 - Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs (ICML 2023).pdf"),
    ("2204.14198", "[12] Flamingo - A Visual Language Model for Few-Shot Learning (NeurIPS 2022).pdf"),
    ("2105.02507", "[14] ChineseFoodNet - A Large-scale Chinese Food Dataset (2021).pdf"),
    ("1606.05675", "[16] Food Image Recognition Using Very Deep Convolutional Networks (BMVC 2016).pdf"),
    ("2210.03629", "[18] ReAct - Synergizing Reasoning and Acting in Language Models (ICLR 2023).pdf"),
    ("2105.05409", "[19] FoodSeg103 - Large-scale Food Dataset and Food Segmentation (TPAMI 2023).pdf"),
    ("1512.03385", "[21] ResNet - Deep Residual Learning for Image Recognition (CVPR 2016).pdf"),
    ("1506.01497", "[22] Faster R-CNN - Towards Real-Time Object Detection with Region Proposal Networks (NeurIPS 2015).pdf"),
    ("1610.02391", "[23] Grad-CAM - Visual Explanations from Deep Networks via Gradient-based Localization (ICCV 2017).pdf"),
    ("1703.05175", "[24] Prototypical Networks for Few-Shot Learning (NeurIPS 2017).pdf"),
    ("1907.01341", "[27] MiDaS - Towards Robust Monocular Depth Estimation (TPAMI 2022).pdf"),
    ("2302.12288", "[28] ZoeDepth - Zero-shot Transfer of a Monocular Dense Depth Model (ICCV 2023).pdf"),
    ("2311.17042", "[29] Advancements in Real-World Food Energy Estimation Using Depth (2023).pdf"),
    ("2409.14003", "[33] Evaluation of GPT-4V and Gemini Advanced in Food Portion Size Estimation (2024).pdf"),
    ("1809.02156", "[35] Object Hallucination in Image Captioning (EMNLP 2018).pdf"),
    ("2305.10355", "[36] POPE - Evaluating Object Hallucination in Large Vision-Language Models (EMNLP 2023).pdf"),
    ("2401.05654", "[37] AMIE - Towards Conversational Diagnostic AI (2024).pdf"),
    ("2304.03442", "[39] Generative Agents - Interactive Simulacra of Human Behavior (UIST 2023).pdf"),
    ("2302.04761", "[40] Toolformer - Language Models Can Teach Themselves to Use Tools (NeurIPS 2023).pdf"),
]

HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def main():
    ok, fail = [], []
    for aid, fname in PAPERS:
        url = f"https://arxiv.org/pdf/{aid}"
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if data[:5] == b"%PDF-":
                with open(fname, "wb") as f:
                    f.write(data)
                ok.append((aid, len(data)))
                print(f"OK   {aid}  {len(data)//1024} KB  -> {fname}")
            else:
                fail.append((aid, "not a PDF (probably unavailable)"))
                print(f"FAIL {aid}  not a PDF ({len(data)} bytes)")
        except Exception as e:
            fail.append((aid, str(e)))
            print(f"FAIL {aid}  {e}")
        time.sleep(3)  # be polite to arXiv
    print(f"\nDone: {len(ok)} ok, {len(fail)} failed")
    for aid, why in fail:
        print(f"  failed: {aid}  {why}")


if __name__ == "__main__":
    main()
