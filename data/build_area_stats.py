'''
data/build_area_stats.py — 预计算每类桶的面积比统计（给分量估计做标定）
================================================
为什么需要：
  旧几何法用固定 plate_ref_px=900 做绝对标定，cm_per_px 与图像分辨率耦合：
  1440px 大图被当 600px 处理 → 食物面积爆高（回锅肉估到 1683g）。
  诊断（diag_ar.py）证实：area_ratio 与短边几乎不相关(corr=0.007)，本身已是
  尺寸归一化的好信号；而 food_px=ar×总像素 与总像素强相关(corr=0.895)。
  故弃用绝对 cm 标定，改用 area_ratio 作"相对调制器"，以类先验为锚。

标定方式：
  对训练集每张图算 area_ratio，按"先验桶"(staple/dish/snack/light)统计中位数。
  估计时：weight = prior × clamp(ar / median_ar_bucket, [lo, hi])。
  这样：① 与图像分辨率解耦；② 以类先验为锚，避免 2D 面积法系统性高估；
        ③ area_ratio 提供有据可依的相对涨落。
输出：data/area_ratio_stats.csv  列 = bucket, median_ar, mean_ar, n
'''
import os
import sys
import numpy as np
import pandas as pd

# 以 `python data/build_area_stats.py` 方式启动时，Python 只把 data/ 加进 sys.path，
# 找不到根目录下的 models/tools 包。补进项目根目录，使直接运行与 -m 方式都能工作。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config
from tools.utils import imread_unicode, food_mask_by_saliency, area_ratio
from data.build_labels import bucket_of, PORTION_PRIOR


def main():
    cfg = load_config()
    data_dir = os.path.join(ROOT, cfg["project"]["data_dir"])
    df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    nutr = pd.read_csv(os.path.join(ROOT, cfg["nutrition"]["db_path"]))
    name_map = dict(zip(nutr["idx"], nutr["zh"]))

    # 训练集很大（1500 张），每类抽 5 张算面积比。
    # GrabCut 单张约 1s，250 张约 4~5 分钟可接受；中位数对桶级标定足够稳。
    sample = df.groupby("label").head(5).reset_index(drop=True)
    print(f"采样 {len(sample)} 张计算面积比...", flush=True)

    records = []
    for i, r in sample.iterrows():
        rel = str(r["path"]).replace("\\", "/")
        p = os.path.join(data_dir, rel)
        img = imread_unicode(p)
        if img is None:
            continue
        m = food_mask_by_saliency(img)
        ar = area_ratio(m)
        b = bucket_of(int(r["label"]), name_map[int(r["label"])])
        records.append({"bucket": b, "ar": ar})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sample)}", flush=True)

    df_ar = pd.DataFrame(records)
    stats = df_ar.groupby("bucket")["ar"].agg(["median", "mean", "count"]).reset_index()
    stats.columns = ["bucket", "median_ar", "mean_ar", "n"]
    out = os.path.join(ROOT, "data", "area_ratio_stats.csv")
    stats.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n-> {out}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
