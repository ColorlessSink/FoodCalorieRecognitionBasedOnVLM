'''
experiments/ablation_study.py — 消融实验
================================================
为什么需要：
  大作业第四部分要求"≥2 个消融实验"。消融 = 拿掉某设计，看指标掉多少，
  用来证明每个设计不是摆设而是真有用。本脚本做两个消融，都聚焦分量估计
  （因为 §2.2 的 v1→v2 演进就是本作业最关键的设计决策，最值得消融验证）。

消融 A：面积比调制 vs 纯先验锚
  - full    : weight = μ × clamp(ar/med, [0.8,1.3]) × w_geo + μ × (1-w_geo)   ← 当前方法
  - no_ar   : 直接退先验 weight = μ                                              ← 拿掉面积比
  - no_clamp: modulator = ar/med 不做 clamp                                      ← 拿掉钳制
  - only_ar : 纯几何 weight = μ × (ar/med)（不锚先验）                          ← 拿掉先验锚
  验证：① 面积比调制带来多少增益（full vs no_ar）；
        ② 钳制防止野值的作用（no_clamp 的离群误差应更大）；
        ③ 先验锚防止系统性高估的作用（only_ar 应大幅高估，MAE 爆表）。

消融 B：geo_weight 扫描
  - w_geo ∈ {0.0, 0.1, 0.2, 0.3, 0.5}
  - 验证当前 0.2 是否为甜点，过大过小是否变差。

消融 C（识别）：模板的影响已在 baseline_eval 里测过（3 模板对比），
  这里补一个 few_shot k-shot 扫描 {1,5,10,20}，验证少样本的边际收益。
'''
import os
import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config, load_classes
from models.food_recognizer import FoodRecognizer, gather_from_split
from models.portion_estimator import PortionEstimator


def _norm(p):
    return str(p).replace("\\", "/")


def _load_truth(cfg):
    """读真值表，键为归一路径。"""
    lbl = pd.read_csv(os.path.join(ROOT, cfg["project"]["data_dir"], "test_labels.csv"))
    tw = {_norm(p): w for p, w in zip(lbl["path"], lbl["weight_g_true"])}
    tc = {_norm(p): c for p, c in zip(lbl["path"], lbl["calories_kcal_true"])}
    return tw, tc


# ============================================================
#  消融 A：分量估计各组件的贡献
# ============================================================
def ablation_portion(cfg, paths, labels, names_zh, n=50):
    """对 4 种变体算 MAE/相对误差/离群率。"""
    from data.build_labels import PORTION_PRIOR, bucket_of
    pe = PortionEstimator(cfg=cfg, use_llm=False)
    tw, _ = _load_truth(cfg)
    name_map = dict(zip(range(len(names_zh)), names_zh))

    # 复用 pe.segment 算 area_ratio，避免重复分割
    results = {"full": [], "no_ar": [], "no_clamp": [], "only_ar": []}
    cal_results = {k: [] for k in results}
    sample_idx = list(range(min(n, len(paths))))

    for i in sample_idx:
        p, l = paths[i], labels[i]
        rel = _norm(os.path.relpath(p, os.path.join(ROOT, cfg["project"]["data_dir"])))
        t = tw.get(rel)
        if t is None:
            continue
        mu, _ = PORTION_PRIOR[bucket_of(l, name_map[l])]
        mask, ar, _ = pe.segment(p)
        med = pe._median_ar(l, name_map[l])

        # full: 当前方法（与 estimate_geometric 同口径，这里重算以保证可对照）
        mod = max(0.8, min(1.3, ar / med if med > 1e-6 else 1.0)) if ar >= 0.05 else 1.0
        w_full = 0.2 * (mu * mod) + 0.8 * mu
        # no_ar: 拿掉面积比，纯先验
        w_no_ar = mu
        # no_clamp: 拿掉 clamp
        mod_nc = ar / med if (med > 1e-6 and ar >= 0.05) else 1.0
        w_no_clamp = 0.2 * (mu * mod_nc) + 0.8 * mu
        # only_ar: 拿掉先验锚，纯几何
        w_only_ar = mu * (ar / med if med > 1e-6 else 1.0) if ar >= 0.05 else mu

        for key, w in [("full", w_full), ("no_ar", w_no_ar),
                       ("no_clamp", w_no_clamp), ("only_ar", w_only_ar)]:
            results[key].append(abs(w - t))
            cal_results[key].append(abs(w - t) / t)

    out = {}
    for key in results:
        e = np.array(results[key]); r = np.array(cal_results[key])
        out[key] = {
            "mae_g": round(float(e.mean()), 1),
            "median_ae_g": round(float(np.median(e)), 1),
            "rel_err_pct": round(float(r.mean())*100, 1),
            "p90_err_g": round(float(np.percentile(e, 90)), 1),   # 离群尾巴
            "le30g_ratio_pct": round(float((e <= 30).mean())*100, 1),
            "n": int(len(e)),
        }
    return out


# ============================================================
#  消融 B：geo_weight 扫描
# ============================================================
def ablation_geo_weight(cfg, paths, labels, names_zh, n=50):
    from data.build_labels import PORTION_PRIOR, bucket_of
    pe = PortionEstimator(cfg=cfg, use_llm=False)
    tw, _ = _load_truth(cfg)
    name_map = dict(zip(range(len(names_zh)), names_zh))

    # 预算 area_ratio
    ars = []
    for i in range(min(n, len(paths))):
        _, ar, _ = pe.segment(paths[i])
        ars.append(ar)

    out = {}
    sample_idx = list(range(min(n, len(paths))))
    for w_geo in [0.0, 0.1, 0.2, 0.3, 0.5]:
        errs = []
        for j, i in enumerate(sample_idx):
            rel = _norm(os.path.relpath(paths[i], os.path.join(ROOT, cfg["project"]["data_dir"])))
            t = tw.get(rel)
            if t is None:
                continue
            l, name = labels[i], name_map[labels[i]]
            mu, _ = PORTION_PRIOR[bucket_of(l, name)]
            med = pe._median_ar(l, name)
            ar = ars[j]
            mod = max(0.8, min(1.3, ar / med if med > 1e-6 else 1.0)) if ar >= 0.05 else 1.0
            w = w_geo * (mu * mod) + (1 - w_geo) * mu
            errs.append(abs(w - t))
        e = np.array(errs)
        out[f"w_geo={w_geo}"] = {
            "mae_g": round(float(e.mean()), 1),
            "median_ae_g": round(float(np.median(e)), 1),
            "le30g_ratio_pct": round(float((e <= 30).mean())*100, 1),
        }
    return out


# ============================================================
#  消融 C：少样本 k-shot 扫描
# ============================================================
def ablation_kshot(cfg, paths, labels, names_zh):
    import random
    tr_paths, tr_labels = gather_from_split(cfg["project"]["data_dir"], "train")
    out = {}
    for k in [1, 5, 10, 20]:
        random.seed(cfg["recognition"]["few_shot"]["seed"])
        sup_paths, sup_labels = [], []
        for idx in load_classes(cfg["project"]["data_dir"])[1]:
            cls_paths = [p for p, l in zip(tr_paths, tr_labels) if l == idx][:k]
            sup_paths += cls_paths
            sup_labels += [idx] * len(cls_paths)
        rec = FoodRecognizer(cfg=cfg, mode="few_shot")
        rec.build_prototypes(sup_paths, sup_labels)
        preds, confs, topk = rec.recognize(paths)
        labels_t = __import__("torch").tensor(labels)
        preds_t = __import__("torch").tensor(preds)
        top1 = (preds_t == labels_t).float().mean().item()
        top5 = sum(labels[i] in [t[0] for t in topk[i][:5]] for i in range(len(labels))) / len(labels)
        out[f"k={k}"] = {"top1": round(top1*100, 2), "top5": round(top5*100, 2), "n_support": len(sup_paths)}
        print(f"  [few_shot] k={k:>2}  Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%  (支持集 {len(sup_paths)} 张)")
    return out


def main():
    cfg = load_config()
    data_dir = cfg["project"]["data_dir"]
    paths, labels = gather_from_split(data_dir, "test")
    names_zh, _, _ = load_classes(data_dir)
    print(f"[数据] test={len(paths)} 张 {len(names_zh)} 类\n")

    print("==== 消融 A：分量估计各组件贡献（n=50）====")
    a = ablation_portion(cfg, paths, labels, names_zh, n=50)
    for k, v in a.items():
        print(f"  {k:<10} MAE={v['mae_g']:>5.1f}g  中位={v['median_ae_g']:>5.1f}g  "
              f"p90={v['p90_err_g']:>5.1f}g  ≤30g占比={v['le30g_ratio_pct']}%")

    print("\n==== 消融 B：geo_weight 扫描（n=50）====")
    b = ablation_geo_weight(cfg, paths, labels, names_zh, n=50)
    for k, v in b.items():
        print(f"  {k:<12} MAE={v['mae_g']:>5.1f}g  中位={v['median_ae_g']:>5.1f}g  ≤30g占比={v['le30g_ratio_pct']}%")

    print("\n==== 消融 C：少样本 k-shot 扫描（全 test）====")
    c = ablation_kshot(cfg, paths, labels, names_zh)

    out = {"ablation_A_components": a, "ablation_B_geo_weight": b, "ablation_C_kshot": c}
    with open(os.path.join(ROOT, "results", "ablation_study.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] results/ablation_study.json")


if __name__ == "__main__":
    main()
