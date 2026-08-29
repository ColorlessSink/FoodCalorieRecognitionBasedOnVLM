'''
experiments/baseline_eval.py — 基线对比与核心指标评估
================================================
为什么需要：
  大作业第四部分要求"基线对比实验（≥2 种 VLM/分量方法）+ 跨场景泛化 +
  卡路里 MAE"。本脚本一次性产出识别/分量/卡路里三类指标，喂给可视化与报告。

产出（写入 results/）：
  1. recognition_baseline.json   : 3 模板 × zero_shot + few_shot + lora_50 的 Top-1/Top-5
  2. method_compare.json         : 精简的方法对比（给可视化用）
  3. portion_calorie_eval.json   : 几何法/CoT 法的 分量 MAE + 卡路里 MAE + 相对误差
  4. scene_eval.json             : standard/real/challenge 三场景下识别与分量表现
'''
import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config, load_classes
from models.food_recognizer import FoodRecognizer, gather_from_split


def eval_recognition(rec, paths, labels):
    """返回 top1, top5, confs, preds。"""
    preds, confs, topk_list = rec.recognize(paths)
    labels_t = torch.tensor(labels)
    preds_t = torch.tensor(preds)
    top1 = (preds_t == labels_t).float().mean().item()
    top5 = sum(labels[i] in [t[0] for t in topk_list[i][:5]] for i in range(len(labels))) / len(labels)
    return top1, top5, confs, preds


def _norm(p):
    """统一路径键为正斜杠字符串，供跨 CSV 匹配。
    test.csv/test_scene.csv 用反斜杠，test_labels.csv 用正斜杠，不归一就会全 miss。"""
    return str(p).replace("\\", "/")


def _stats(errs, rels, cal_errs, cal_rels):
    """统一的误差统计。errs/rels/cal_errs/cal_rels 为 list。"""
    errs, rels = np.array(errs), np.array(rels)
    cal_errs, cal_rels = np.array(cal_errs), np.array(cal_rels)
    return {
        "mae_g": round(float(errs.mean()), 1),
        "median_ae_g": round(float(np.median(errs)), 1),
        "rel_err_pct": round(float(rels.mean())*100, 1),
        "le30g_ratio_pct": round(float((errs <= 30).mean())*100, 1),
        "calorie_mae_kcal": round(float(cal_errs.mean()), 1),
        "calorie_rel_pct": round(float(cal_rels.mean())*100, 1),
        "n": int(len(errs)),
    }


def run_recognition_baselines(cfg, paths, labels, names_zh):
    """3 种模板的零样本对比 + few_shot + lora_50。"""
    results = []
    templates = cfg["recognition"]["templates_for_compare"]
    rec = None
    # 3 模板 zero_shot
    for tpl in templates:
        rec = FoodRecognizer(cfg=cfg, mode="zero_shot")
        rec.build_text_feats(template=tpl)
        t0 = time.time()
        top1, top5, confs, _ = eval_recognition(rec, paths, labels)
        dt = time.time() - t0
        results.append({
            "method": "zero_shot", "template": tpl,
            "top1": round(top1*100, 2), "top5": round(top5*100, 2),
            "mean_conf": round(float(np.mean(confs)), 3), "seconds": round(dt, 1),
        })
        print(f"  [zero_shot] {tpl}  Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%  ({dt:.1f}s)")

    # few_shot 10-shot
    # 方法学洁净：支持集从 train split 取前 k 张/类，避免用测试图当支持（数据泄漏）。
    import random
    random.seed(cfg["recognition"]["few_shot"]["seed"])
    k = cfg["recognition"]["few_shot"]["k_shot"]
    tr_paths, tr_labels = gather_from_split(cfg["project"]["data_dir"], "train")
    sup_paths, sup_labels = [], []
    for idx in rec.class_idx:
        cls_paths = [p for p, l in zip(tr_paths, tr_labels) if l == idx][:k]
        sup_paths += cls_paths
        sup_labels += [idx] * len(cls_paths)
    rec_fs = FoodRecognizer(cfg=cfg, mode="few_shot")
    rec_fs.build_prototypes(sup_paths, sup_labels)
    top1, top5, confs, _ = eval_recognition(rec_fs, paths, labels)
    results.append({"method": "few_shot", "k_shot": k,
                    "top1": round(top1*100, 2), "top5": round(top5*100, 2),
                    "mean_conf": round(float(np.mean(confs)), 3)})
    print(f"  [few_shot]  k={k}  Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%")

    # lora 50（adapter 存在才评）
    adapter_dir = os.path.join(ROOT, cfg["recognition"]["lora"]["adapter_dir_50"])
    if os.path.isdir(adapter_dir) and os.listdir(adapter_dir):
        rec_lora = FoodRecognizer(cfg=cfg, mode="lora")
        top1, top5, confs, _ = eval_recognition(rec_lora, paths, labels)
        results.append({"method": "lora_50",
                        "top1": round(top1*100, 2), "top5": round(top5*100, 2),
                        "mean_conf": round(float(np.mean(confs)), 3)})
        print(f"  [lora_50]   Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%")
    else:
        print("  [lora_50] 跳过：adapter 未训练（先跑 experiments/train_lora_50.py）")
    return results


def run_portion_calorie_eval(cfg, paths, labels, names_zh, n=30):
    """几何法 vs CoT 法的分量/卡路里 MAE。CoT 慢，只抽 n 张。"""
    from models.portion_estimator import PortionEstimator
    from models.nutrition_querier import NutritionQuerier

    lbl_df = pd.read_csv(os.path.join(ROOT, cfg["project"]["data_dir"], "test_labels.csv"))
    truth_w = {_norm(p): w for p, w in zip(lbl_df["path"], lbl_df["weight_g_true"])}
    truth_c = {_norm(p): c for p, c in zip(lbl_df["path"], lbl_df["calories_kcal_true"])}

    pe_geo = PortionEstimator(cfg=cfg, use_llm=False)
    nq = NutritionQuerier(cfg=cfg)
    pe_cot = PortionEstimator(cfg=cfg, use_llm=True)

    geo_err, geo_cal_err, geo_rel, geo_cal_rel = [], [], [], []
    cot_err, cot_cal_err, cot_rel, cot_cal_rel = [], [], [], []
    name_map = dict(zip(range(len(names_zh)), names_zh))

    sample_idx = list(range(min(n, len(paths))))
    for i in sample_idx:
        p, l = paths[i], labels[i]
        # 路径键归一：test.csv 是反斜杠，test_labels.csv 是正斜杠
        rel = _norm(os.path.relpath(p, os.path.join(ROOT, cfg["project"]["data_dir"])))
        t_w = truth_w.get(rel)
        t_c = truth_c.get(rel)
        if t_w is None:
            continue
        name = name_map[l]

        # 几何
        g = pe_geo.estimate_geometric(p, l, name)
        gw = g["weight_g"]
        g_cal = nq.compute(l, gw)["kcal"]
        geo_err.append(abs(gw - t_w)); geo_rel.append(abs(gw - t_w)/t_w)
        geo_cal_err.append(abs(g_cal - (t_c or 0))); geo_cal_rel.append(abs(g_cal-(t_c or 0))/max(t_c,1))

        # CoT
        c = pe_cot.estimate_cot(p, l, name)
        cw = c["weight_g"]
        c_cal = nq.compute(l, cw)["kcal"]
        cot_err.append(abs(cw - t_w)); cot_rel.append(abs(cw - t_w)/t_w)
        cot_cal_err.append(abs(c_cal - (t_c or 0))); cot_cal_rel.append(abs(c_cal-(t_c or 0))/max(t_c,1))

        if (i+1) % 5 == 0:
            print(f"  已评 {i+1}/{len(sample_idx)} 张 (geo_mae={np.mean(geo_err):.1f}g cot_mae={np.mean(cot_err):.1f}g)")

    return {
        "geometric": _stats(geo_err, geo_rel, geo_cal_err, geo_cal_rel),
        "cot_llm": _stats(cot_err, cot_rel, cot_cal_err, cot_cal_rel),
    }


def run_scene_eval(cfg, paths, labels, names_zh):
    """三场景下的识别 Top-1 + 几何法分量 MAE。"""
    scene_df = pd.read_csv(os.path.join(ROOT, cfg["project"]["data_dir"], "test_scene.csv"))
    scene_map = {_norm(p): s for p, s in zip(scene_df["path"], scene_df["scene"])}

    lbl_df = pd.read_csv(os.path.join(ROOT, cfg["project"]["data_dir"], "test_labels.csv"))
    truth = {_norm(p): w for p, w in zip(lbl_df["path"], lbl_df["weight_g_true"])}

    from models.portion_estimator import PortionEstimator
    pe = PortionEstimator(cfg=cfg, use_llm=False)
    name_map = dict(zip(range(len(names_zh)), names_zh))

    out = {}
    for sc in ["standard", "real", "challenge"]:
        idxs = []
        for i, p in enumerate(paths):
            rel = _norm(os.path.relpath(p, os.path.join(ROOT, cfg["project"]["data_dir"])))
            if scene_map.get(rel) == sc:
                idxs.append(i)
        if not idxs:
            continue
        sc_paths = [paths[i] for i in idxs]
        sc_labels = [labels[i] for i in idxs]
        rec = FoodRecognizer(cfg=cfg, mode="zero_shot")
        top1, top5, _, preds = eval_recognition(rec, sc_paths, sc_labels)

        errs = []
        for p, l in zip(sc_paths, sc_labels):
            rel = _norm(os.path.relpath(p, os.path.join(ROOT, cfg["project"]["data_dir"])))
            t = truth.get(rel)
            if t is None:
                continue
            g = pe.estimate_geometric(p, l, name_map[l])
            errs.append(abs(g["weight_g"] - t))
        errs = np.array(errs) if errs else np.array([0])
        out[sc] = {
            "n": len(idxs),
            "top1": round(top1*100, 2),
            "top5": round(top5*100, 2),
            "portion_mae_g": round(float(errs.mean()), 1),
        }
        print(f"  [{sc}] n={len(idxs)} top1={top1*100:.2f}% portion_mae={errs.mean():.1f}g")
    return out


def main():
    cfg = load_config()
    data_dir = cfg["project"]["data_dir"]
    paths, labels = gather_from_split(data_dir, "test")
    names_zh, _, _ = load_classes(data_dir)
    print(f"[数据] {data_dir} test={len(paths)} 张 {len(names_zh)} 类\n")

    # ---- 1. 识别基线（3 模板 + few_shot + lora_50）----
    print("==== 1. 识别基线对比 ====")
    rec_results = run_recognition_baselines(cfg, paths, labels, names_zh)
    with open(os.path.join(ROOT, "results", "recognition_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(rec_results, f, ensure_ascii=False, indent=2)
    method_compare = [{"method": r.get("method"), "template": r.get("template",""),
                       "top1": r["top1"], "top5": r["top5"]} for r in rec_results]
    with open(os.path.join(ROOT, "results", "method_compare.json"), "w", encoding="utf-8") as f:
        json.dump(method_compare, f, ensure_ascii=False, indent=2)

    # ---- 2. 分量/卡路里 MAE（几何 vs CoT）----
    print("\n==== 2. 分量/卡路里评估（几何 vs CoT）====")
    n_cot = 30
    pc = run_portion_calorie_eval(cfg, paths, labels, names_zh, n=n_cot)
    with open(os.path.join(ROOT, "results", "portion_calorie_eval.json"), "w", encoding="utf-8") as f:
        json.dump(pc, f, ensure_ascii=False, indent=2)
    print(f"  几何法 MAE={pc['geometric']['mae_g']}g  卡路里MAE={pc['geometric']['calorie_mae_kcal']}kcal")
    print(f"  CoT法  MAE={pc['cot_llm']['mae_g']}g  卡路里MAE={pc['cot_llm']['calorie_mae_kcal']}kcal")

    # ---- 3. 跨场景 ----
    print("\n==== 3. 跨场景泛化 ====")
    scene = run_scene_eval(cfg, paths, labels, names_zh)
    with open(os.path.join(ROOT, "results", "scene_eval.json"), "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)

    print("\n[完成] 全部结果已写入 results/")


if __name__ == "__main__":
    main()
