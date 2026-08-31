'''
外部抓取数据的评估（接入 dataset_external/，不碰 dataset_50cls/ 的评估口径）
---
评估三层（与内部 experiments/ 对齐口径）：
  ① 识别层：500 张外部单食物图，zero_shot / few_shot / lora 三模式 Top-1/Top-5
     - 这是"域泛化"检验：内部 test 是 ChineseFoodNet 同源，外部是百度/Bing 真实网图，
        分布更接近用户实际拍照，比内部数字更能说明上线水平。
  ② 分量/卡路里层：合成真值（data/build_external_labels.py，与内部同 σ 口径）
     - 分量 MAE（oracle，几何法）+ 卡路里 MAE（oracle / e2e 两口径）。
     分量几何法有 GrabCut 逐张分割（CPU 慢），按 3 张/类抽样 150 张跑，识别仍全量。
  ③ 混合盘：40 张真实混合菜图无 GT（网图，没有"哪几样菜各多少克"标注），
     只能做描述性统计（区域数分布 / 预标类别分布 / 置信度），并如实记录
     "检测器在真实照片上 37/40 整图当一块"的域差，不给虚张的 MAE。

产出：results/external_eval.json
'''
import os, sys, json, random
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config, load_classes
from models.food_recognizer import FoodRecognizer, gather_from_split
from tools.utils import imread_unicode

EXT = os.path.join(ROOT, "dataset_external")


def build_recognizer(cfg, mode, data_dir):
    # 与 mixed_eval.py / label_mixed.py 同口径地构建识别器
    if mode == "few_shot":
        random.seed(cfg["recognition"]["few_shot"]["seed"])
        k = cfg["recognition"]["few_shot"]["k_shot"]
        tr_paths, tr_labels = gather_from_split(data_dir, "train")
        sup_paths, sup_labels = [], []
        for idx in load_classes(data_dir)[1]:
            cls_paths = [p for p, l in zip(tr_paths, tr_labels) if l == idx][:k]
            sup_paths += cls_paths
            sup_labels += [idx] * len(cls_paths)
        rec = FoodRecognizer(cfg=cfg, mode="few_shot", data_dir=data_dir)
        rec.build_prototypes(sup_paths, sup_labels)
        return rec
    return FoodRecognizer(cfg=cfg, mode=mode, data_dir=data_dir)


def eval_recognition(cfg, df, data_dir):
    names_zh, _, _ = load_classes(data_dir)
    paths = [os.path.join(EXT, p) for p in df["path"]]
    labels = df["label"].tolist()
    n = len(paths)

    out = {}
    modes = ["zero_shot", "few_shot", "lora"]
    adapter = os.path.join(ROOT, cfg["recognition"]["lora"]["adapter_dir_50"])
    for mode in modes:
        if mode == "lora" and not (os.path.isdir(adapter) and os.listdir(adapter)):
            out[mode] = {"skipped": "无 LoRA adapter（results/lora_adapter_50 不存在），跳过"}
            print(f"  [lora] 跳过：无 adapter")
            continue
        rec = build_recognizer(cfg, mode, data_dir)
        preds, confs, topk_list = rec.recognize(paths)
        top1 = sum(p == l for p, l in zip(preds, labels)) / n
        top5 = sum(labels[i] in [t[0] for t in topk_list[i][:5]]
                   for i in range(n)) / n
        out[mode] = {
            "top1_pct": round(top1 * 100, 2),
            "top5_pct": round(top5 * 100, 2),
            "mean_conf": round(float(np.mean(confs)), 3),
        }
        print(f"  [{mode:>9}] Top-1 {top1*100:5.2f}%  Top-5 {top5*100:5.2f}%  "
              f"mean_conf {np.mean(confs):.3f}")
    return out


def eval_portion_calorie(cfg, df, data_dir, sample_per_cls=3):
    # 分量 + 卡路里：oracle 口径（GT 类别）+ e2e 口径（few_shot 识别类别）
    from models.portion_estimator import PortionEstimator
    from models.nutrition_querier import NutritionQuerier
    names_zh, _, _ = load_classes(data_dir)
    name_map = dict(zip(load_classes(data_dir)[1], names_zh))

    # 分层抽样：每类 sample_per_cls 张，固定 seed 可复现
    rng = random.Random(67)
    sample_idx = []
    for label, g in df.groupby("label"):
        idxs = list(g.index)
        sample_idx += rng.sample(idxs, min(sample_per_cls, len(idxs)))
    sdf = df.loc[sample_idx].reset_index(drop=True)

    pe = PortionEstimator(cfg=cfg, use_llm=False)
    nq = NutritionQuerier(cfg=cfg)
    # 识别类别用 few_shot（内部混合盘默认、单食物上也最高 83.67%）
    rec = build_recognizer(cfg, "few_shot", data_dir)
    paths = [os.path.join(EXT, p) for p in sdf["path"]]
    preds, _, _ = rec.recognize(paths)

    w_errs, kcal_oracle_errs, kcal_e2e_errs = [], [], []
    for i, row in sdf.iterrows():
        idx = int(row["label"])
        w = pe.estimate_geometric(paths[i], idx, name_map[idx])["weight_g"]
        w_errs.append(abs(w - row["weight_g_true"]))
        kcal_oracle_errs.append(abs(nq.compute(idx, w)["kcal"] - row["calories_kcal_true"]))
        kcal_e2e_errs.append(abs(nq.compute(preds[i], w)["kcal"] - row["calories_kcal_true"]))

    w_errs = np.array(w_errs)
    kcal_o = np.array(kcal_oracle_errs)
    kcal_e = np.array(kcal_e2e_errs)
    return {
        "n_sampled": int(len(sdf)),
        "portion_mae_g": round(float(w_errs.mean()), 1),
        "portion_median_g": round(float(np.median(w_errs)), 1),
        "portion_rel_pct": round(float((w_errs / sdf["weight_g_true"]).mean() * 100), 1),
        "calorie_oracle_mae_kcal": round(float(kcal_o.mean()), 1),
        "calorie_e2e_mae_kcal": round(float(kcal_e.mean()), 1),
        "calorie_e2e_median_kcal": round(float(np.median(kcal_e)), 1),
    }


def summarize_mixed():
    # 真实混合菜图：读已生成的机器预标 CSV，做描述性统计（无 GT，不给 MAE）
    csv_path = os.path.join(EXT, "mixed_labels.csv")
    if not os.path.exists(csv_path):
        return {"skipped": "mixed_labels.csv 不存在，请先跑 tools/label_mixed.py"}
    df = pd.read_csv(csv_path)
    per_plate = df.groupby("plate")["comp"].count()
    n_plates = df["plate"].nunique()
    region_dist = per_plate.value_counts().sort_index().to_dict()
    conf = df["conf"]
    names_zh, _, _ = load_classes(load_config()["project"]["data_dir"])
    df["label_zh"] = df["label"].apply(lambda i: names_zh[int(i)])
    top_zh = df["label_zh"].value_counts().head(8).to_dict()
    multi = per_plate[per_plate > 1].index.tolist()
    return {
        "n_plates": int(n_plates),
        "n_component_rows": int(len(df)),
        "region_count_dist": {str(k): int(v) for k, v in region_dist.items()},
        "single_region_plates": int((per_plate == 1).sum()),
        "multi_region_plates": multi,
        "high_conf_ge0.8": int((conf >= 0.8).sum()),
        "low_conf_lt0.3": int((conf < 0.3).sum()),
        "top_predicted_dishes": {k: int(v) for k, v in top_zh.items()},
        "note": "真实混合菜无 GT（网图无分量/类别标注），机器预标仅作展示与人工复核起点，"
                "不给虚张的 MAE。检测器在真实照片上 37/40 整图当一块（域差，见 process_log）。",
    }


def main():
    cfg = load_config()
    data_dir = cfg["project"]["data_dir"]
    df = pd.read_csv(os.path.join(EXT, "single_labels_ext.csv"))
    print(f"外部单食物 {len(df)} 张 / {df['label'].nunique()} 类\n")

    out = {}
    print("[1/3] 识别（全量 500 张，三模式）...")
    out["recognition"] = eval_recognition(cfg, df, data_dir)

    print("\n[2/3] 分量 + 卡路里（分层抽样，oracle / e2e 两口径）...")
    pc = eval_portion_calorie(cfg, df, data_dir)
    out["portion_calorie"] = pc
    print(f"  分量 MAE {pc['portion_mae_g']}g（中位 {pc['portion_median_g']}g，"
          f"相对 {pc['portion_rel_pct']}%）")
    print(f"  卡路里 oracle {pc['calorie_oracle_mae_kcal']}kcal / "
          f"e2e {pc['calorie_e2e_mae_kcal']}kcal（中位 {pc['calorie_e2e_median_kcal']}kcal）")

    print("\n[3/3] 真实混合菜描述性统计...")
    mx = summarize_mixed()
    out["mixed_plates"] = mx
    if "skipped" in mx:
        print(f"  {mx['skipped']}")
    else:
        print(f"  40 张里 {mx['single_region_plates']} 张单区域、"
              f"{len(mx['multi_region_plates'])} 张多区域 {mx['multi_region_plates']}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out_path = os.path.join(ROOT, "results", "external_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n-> {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
