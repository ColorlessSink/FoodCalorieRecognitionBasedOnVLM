'''
experiments/failure_analysis.py — 失败案例分析
================================================
为什么需要：
  大作业第四部分要求"≥15 个失败案例分析"。失败案例比成功案例更有信息量——
  它揭示模型的边界：哪些类易混、哪种拍摄条件下识别/分量会崩。本脚本自动从
  test 结果里捞出失败样本并归类，再人工补充原因分析写入 report/failure_cases.md。

产出：
  results/failure_cases_raw.json  : 自动捞取的失败样本（识别错 + 分量误差大）
  report/failure_cases.md         : 人工归因的 ≥15 个典型案例（含图、原因、改进方向）

失败判据：
  - 识别失败：pred != label
  - 分量大误差：|weight - true| > 60g（2× 硬指标，筛显著错）
  - 低置信：conf < 0.3（门控会触发反问的样本）
'''
import os
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config, load_classes
from models.food_recognizer import FoodRecognizer, gather_from_split
from models.portion_estimator import PortionEstimator


def _norm(p):
    return str(p).replace("\\", "/")


def main():
    cfg = load_config()
    data_dir = cfg["project"]["data_dir"]
    paths, labels = gather_from_split(data_dir, "test")
    names_zh, _, _ = load_classes(data_dir)
    name_map = dict(zip(range(len(names_zh)), names_zh))

    lbl = pd.read_csv(os.path.join(ROOT, data_dir, "test_labels.csv"))
    tw = {_norm(p): w for p, w in zip(lbl["path"], lbl["weight_g_true"])}
    tc = {_norm(p): c for p, c in zip(lbl["path"], lbl["calories_kcal_true"])}

    # 识别：用 LoRA（当前最佳）跑全 test
    print("==== 识别失败（LoRA_50 全 test）====")
    rec = FoodRecognizer(cfg=cfg, mode="lora")
    preds, confs, topk = rec.recognize(paths)

    recog_fails = []
    confus_pairs = {}   # (true,pred) -> count
    for i in range(len(labels)):
        if preds[i] != labels[i]:
            recog_fails.append({
                "path": _norm(paths[i]), "true": name_map[labels[i]], "true_idx": labels[i],
                "pred": name_map[preds[i]], "pred_idx": preds[i],
                "conf": round(confs[i], 3),
                "top3": [(t[1], round(t[2], 3)) for t in topk[i][:3]],
            })
            key = f"{name_map[labels[i]]} -> {name_map[preds[i]]}"
            confus_pairs[key] = confus_pairs.get(key, 0) + 1
    print(f"  识别错误 {len(recog_fails)}/{len(labels)} ({len(recog_fails)/len(labels)*100:.1f}%)")
    print(f"  最易混淆 Top-8：")
    for k, v in sorted(confus_pairs.items(), key=lambda x: -x[1])[:8]:
        print(f"    {k}  ({v} 次)")

    # 分量大误差（几何法，全 test）
    # 注：全 600 张逐张 GrabCut 分割太慢（每张 1-2s），这里抽 200 张做
    # 失败案例捞取——取每 3 张一张，保证类分布与全量一致。
    print("\n==== 分量大误差（几何法 抽样 200/600）====")
    pe = PortionEstimator(cfg=cfg, use_llm=False)
    portion_fails = []
    portion_sample_idx = list(range(0, len(paths), 3))  # 每3张取1张 ≈ 200张
    sampled = 0
    for i in portion_sample_idx:
        p, l = paths[i], labels[i]
        rel = _norm(os.path.relpath(p, os.path.join(ROOT, data_dir)))
        t = tw.get(rel)
        if t is None:
            continue
        sampled += 1
        g = pe.estimate_geometric(p, l, name_map[l])
        e = abs(g["weight_g"] - t)
        if e > 60:
            portion_fails.append({
                "path": rel, "food": name_map[l],
                "est": g["weight_g"], "true": round(t, 1),
                "err": round(e, 1), "area_ratio": g["area_ratio"],
                "modulator": g.get("modulator", 1.0),
            })
        if sampled % 50 == 0:
            print(f"  [进度] 已处理 {sampled}/{len(portion_sample_idx)} 张")
    portion_fails.sort(key=lambda x: -x["err"])
    print(f"  抽样 {sampled} 张，分量误差>60g 共 {len(portion_fails)} 张")
    for f in portion_fails[:10]:
        print(f"    {f['food']:<8} est={f['est']:>6.1f} true={f['true']:>6.1f} err={f['err']:>5.1f} ar={f['area_ratio']:.3f}")

    # 低置信样本
    low_conf = [r for r in recog_fails if r["conf"] < 0.3]
    print(f"\n==== 低置信(<0.3)失败样本 {len(low_conf)} ====")

    out = {
        "recognition_fails": recog_fails,
        "confusion_pairs_top": dict(sorted(confus_pairs.items(), key=lambda x: -x[1])[:15]),
        "portion_big_err": portion_fails,
        "low_confidence_fails": low_conf,
        "summary": {
            "recog_err_count": len(recog_fails),
            "recog_err_rate": round(len(recog_fails)/len(labels)*100, 2),
            "portion_big_err_count": len(portion_fails),
        },
    }
    out_path = os.path.join(ROOT, "results", "failure_cases_raw.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] {out_path}")


if __name__ == "__main__":
    main()
