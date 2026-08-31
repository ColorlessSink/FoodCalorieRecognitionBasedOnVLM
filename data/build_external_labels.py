'''
外部抓取单食物图的合成真值生成（接入 dataset_external/，不影响 dataset_50cls/）
---
背景：/goal 要求"自己采集一些图片数据，从网上下载部分图片并标注一下，接入原先项目"。
  tools/scrape_images.py 已从百度/Bing 抓了 500 张单食物图（50 类 × 10 张），
  存到 dataset_external/images/<中文菜名>/<菜名>_NNN.jpg，另有 40 张真实混合菜图。

这里的"标注"分两层：
  ① 类别真值：从文件夹名（中文菜名）→ classes_50.csv 的 idx，无需人工（抓取时就按类抓的）
  ② 分量/卡路里真值：与 dataset_50cls 完全同口径的"合成真值"——文件名 hash + 类先验高斯。
     为什么用合成真值而不是人工称重：网图没有分量标注，且与内部 600 张测试图同口径，
     这样外部图的分量/卡路里评估才能和内部结果直接对比（同一把尺子）。

关键：这里 import data/build_labels.py 的 PORTION_PRIOR / bucket_of / hash_seed /
  gaussian_from_hash，不复制一份——保证外部真值的 σ 口径与内部逐字节一致，
  避免"改了内部先验忘了同步外部"的不一致。

输出：dataset_external/single_labels_ext.csv
  列 = path,label,name,weight_g_true,calories_kcal_true
  path 相对 dataset_external/（如 images/麻婆豆腐/麻婆豆腐_001.jpg）
'''
import os, sys
import pandas as pd

# 运行 `python data/build_external_labels.py` 时 sys.path 只有 data/，补上项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_classes
from data.build_labels import PORTION_PRIOR, bucket_of, hash_seed, gaussian_from_hash

EXT = os.path.join(ROOT, "dataset_external")
IMG_DIR = os.path.join(EXT, "images")
NUTR = os.path.join(ROOT, "data", "nutrition_db.csv")


def main():
    names_zh, idxs, classes_df = load_classes("dataset_50cls")
    zh_to_idx = dict(zip(names_zh, idxs))
    nutr = pd.read_csv(NUTR)
    kcal_map = {int(r["idx"]): float(r["kcal_per100g"]) for _, r in nutr.iterrows()}

    rows = []
    missing = []       # 抓了但不在 50 类里的文件夹（理论不该有，防御性记录）
    for folder in sorted(os.listdir(IMG_DIR)):
        folder_path = os.path.join(IMG_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder not in zh_to_idx:
            missing.append(folder)
            continue
        idx = zh_to_idx[folder]
        files = sorted(f for f in os.listdir(folder_path) if f.endswith(".jpg"))
        mu, sigma = PORTION_PRIOR[bucket_of(idx, folder)]
        for fn in files:
            rel = f"images/{folder}/{fn}"
            u = hash_seed(rel)
            weight = round(gaussian_from_hash(u, mu, sigma), 1)
            kcal = round(weight * kcal_map[idx] / 100, 1)
            rows.append({
                "path": rel, "label": idx, "name": folder,
                "weight_g_true": weight, "calories_kcal_true": kcal,
            })

    out = pd.DataFrame(rows)
    out_path = os.path.join(EXT, "single_labels_ext.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 完整性校验
    n_img = len(out)
    n_cls = out["label"].nunique()
    per_cls = out.groupby("label")["path"].count()
    print(f"外部单食物真值 -> {os.path.relpath(out_path, ROOT)}")
    print(f"  图数 {n_img} | 类别 {n_cls} | 每类 {per_cls.min()}~{per_cls.max()} 张")
    print(f"  重量均值 {out['weight_g_true'].mean():.1f}g | "
          f"卡路里均值 {out['calories_kcal_true'].mean():.1f}kcal")
    if missing:
        print(f"  ⚠️ 以下文件夹不在 50 类里，已跳过：{missing}")


if __name__ == "__main__":
    main()
