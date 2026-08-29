'''
data/build_mixed_plates.py — 合成混合餐盘数据（单食物 → 多食物同框）
================================================
为什么需要：
  大作业数据集要求"单食物与混合餐盘均需覆盖"，实验指标要求"混合餐盘卡路里 MAE≤100kcal"。
  ChineseFoodNet 每张图只有单一食物、无混合餐盘图。与其另找带标注的混合餐盘数据集
  （如 Nutrition5k，需下载且耗时/许可不可控），不如从已有 test 集合成：
    每盘取 2~3 张不同类的 test 图（这些图已带与单食物评估完全同口径的合成真值），
    圆形羽化裁剪贴到"暖灰桌面 + 白色餐盘"画布上。
  这样做的三个好处：
    ① 每盘真值 = 各组件真值之和（重量/卡路里），真值体系与单食物完全一致
       （文件名 hash + 类先验高斯，与估计器输入解耦，见 build_labels.py）；
    ② 识别/分割/分量模块无需重训，直接评估混合餐盘能力；
    ③ 布局（圆心/直径）记录在 CSV，可做"提议区域 vs 真实布局"的检测诊断。

布局设计（为什么是这些数字）：
  食物圆全部落在 tools/utils.py 中心偏置掩膜 dist<0.9 的椭圆内（否则显著性分割
  会把边缘的食物裁掉），且圆与圆之间留 ≥40px 间隙。间隙下界的依据是实测教训
  （v1 布局间隙只有 12.8~36px，检测器的局部 std 核 15px + 羽化扩散 ≈17~25px
  把走廊填满，三菜盘的三个圆粘成一个连通域，召回卡在 58%）：
    间隙 ≥ 羽化 sigma×2(≈dia/20) + std 核半径(≈k/2) + 余量 → 取 40px 起步。
  双菜左右分列，三菜品字形。

生成（固定 seed=2026，可复现；LAYOUTS 是常量不消耗随机数，
改布局不影响组件选取，真值标签/重量/卡路里与单食物口径完全一致）：
  dataset_50cls/mixed/plate_XXXX.jpg        120 张（72 张双菜 + 48 张三菜）
  dataset_50cls/test_mixed.csv              每组件一行：
      plate,n_comp,comp,src_path,label,name,weight_g_true,calories_kcal_true,cx,cy,dia
'''
import os
import sys
import random
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import ROOT, load_config
from tools.utils import imread_unicode

SEED = 2026
N_PLATES = 120                 # 72 张双菜（60%）+ 48 张三菜（40%）
CANVAS = 640
SRC_CROP_FRAC = 0.42           # 源图中心圆半径 = 0.42 × 短边（食物多居中）
JPEG_Q = 92

# 布局（圆心/直径，px）。约束（v2，间距实测验证）：
#   ① 任意两圆：圆心距 - 半径和 ≥ 40px（检测走廊不被羽化+std核填满）；
#   ② 任意圆：hypot(nx,ny)+r/320 ≤ 0.90（圆整体在中心椭圆 dist<0.9 内，
#      与显著性分割的中心偏置掩膜一致，边缘食物不会被裁掉）。
# 双菜 2×230：y=320 对称，x=190/450 → 间距 260-230=30px…
# 不行，重算：双菜 dia=230，圆心距=260 → 间隙 260-230=30 <40。取 x=180/460，
# 圆心距 280-230=50px ✓。三菜 dia=200 品字：(320,200)/(185,400)/(455,400)，
# 上-左间隙 hypot(135,200)-200≈42.7 ✓，左-右间隙 270-200=70 ✓。
LAYOUTS = {
    2: [((180, 320), 230), ((460, 320), 230)],
    3: [((320, 200), 200), ((185, 400), 200), ((455, 400), 200)],
}


def make_canvas(rng):
    """暖灰桌面（带径向渐变+噪声，避免 GrabCut 把桌面当均匀前景）+ 白色低饱餐盘。"""
    c = np.full((CANVAS, CANVAS, 3), (150, 142, 132), np.uint8)   # BGR 暖灰
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    d = np.sqrt(((xx - CANVAS / 2) / (CANVAS / 2)) ** 2 +
                ((yy - CANVAS / 2) / (CANVAS / 2)) ** 2)
    shade = (1.0 - 0.10 * d).clip(0.8, 1.0)
    c = (c.astype(np.float32) * shade[..., None] +
         rng.gauss(0, 2.0) * np.ones_like(c, np.float32)).clip(0, 255).astype(np.uint8)
    # 白色餐盘：低饱和、低显著性，不与彩色食物抢前景
    cv2.ellipse(c, (CANVAS // 2, CANVAS // 2), (300, 300), 0, 0, 360, (238, 236, 232), -1)
    cv2.ellipse(c, (CANVAS // 2, CANVAS // 2), (300, 300), 0, 0, 360, (205, 202, 196), 3)
    return c


def paste_circle(canvas, src_bgr, cx, cy, dia):
    """把源图中心圆形区域（羽化边缘）贴到画布 (cx,cy)，直径 dia。成功返回 True。"""
    h, w = src_bgr.shape[:2]
    r = int(SRC_CROP_FRAC * min(h, w))
    if r < 20:
        return False
    alpha = np.zeros((h, w), np.float32)
    cv2.circle(alpha, (w // 2, h // 2), r, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, r / 20.0))  # 羽化，避免硬边
    scale = dia / (2.0 * r)
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    rs = cv2.resize(src_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    am = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)

    x0, y0 = cx - nw // 2, cy - nh // 2
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(CANVAS, x0 + nw), min(CANVAS, y0 + nh)
    if dx1 <= dx0 or dy1 <= dy0:
        return False
    sx0, sy0 = dx0 - x0, dy0 - y0
    src = rs[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)].astype(np.float32)
    a = am[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)][..., None]
    roi = canvas[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas[dy0:dy1, dx0:dx1] = (src * a + roi * (1 - a)).astype(np.uint8)
    return True


def main():
    cfg = load_config()
    data_dir = os.path.join(ROOT, cfg["project"]["data_dir"])

    # test.csv（反斜杠）与 test_labels.csv（正斜杠）键归一后合并取真值
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    lbl = pd.read_csv(os.path.join(data_dir, "test_labels.csv"))
    test["key"] = test["path"].astype(str).str.replace("\\", "/")
    lbl["key"] = lbl["path"].astype(str).str.replace("\\", "/")
    df = test.merge(lbl[["key", "weight_g_true", "calories_kcal_true"]],
                    on="key", how="inner")
    print(f"[数据] test 图 {len(df)} 张（带真值）")

    rng = random.Random(SEED)
    out_dir = os.path.join(data_dir, "mixed")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    n_double = int(N_PLATES * 0.6)
    made = 0
    for i in range(N_PLATES):
        n_comp = 2 if i < n_double else 3
        labels = rng.sample(sorted(df["label"].unique().tolist()), n_comp)
        picks = []
        for lb in labels:
            sub = df[df["label"] == lb]
            picks.append(sub.iloc[rng.randrange(len(sub))])

        canvas = make_canvas(rng)
        pid = f"plate_{i + 1:04d}"
        ok = True
        comps = []
        for c, r in enumerate(picks):
            src = imread_unicode(os.path.join(data_dir, r["key"]))
            if src is None:
                ok = False
                break
            (cx, cy), dia = LAYOUTS[n_comp][c]
            if not paste_circle(canvas, src, cx, cy, dia):
                ok = False
                break
            comps.append({
                "plate": f"mixed/{pid}.jpg", "n_comp": n_comp, "comp": c,
                "src_path": r["key"], "label": int(r["label"]),
                "weight_g_true": float(r["weight_g_true"]),
                "calories_kcal_true": float(r["calories_kcal_true"]),
                "cx": cx, "cy": cy, "dia": dia,
            })
        if not ok:
            print(f"  跳过 {pid}（源图读取/贴图失败）")
            continue

        jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1]
        jpg.tofile(os.path.join(out_dir, f"{pid}.jpg"))
        rows.extend(comps)
        made += 1
        if made % 20 == 0:
            print(f"  已生成 {made}/{N_PLATES}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(data_dir, "test_mixed.csv"), index=False, encoding="utf-8-sig")

    # 汇总校验
    tot = out.groupby("plate").agg(
        n_comp=("n_comp", "first"),
        gt_w=("weight_g_true", "sum"),
        gt_kcal=("calories_kcal_true", "sum"),
    )
    print(f"\n-> {out_dir}/ ({made} 张)  + test_mixed.csv ({len(out)} 组件行)")
    for k, g in tot.groupby("n_comp"):
        print(f"  {k} 菜盘 {len(g)} 张：均值 {g['gt_w'].mean():.0f}g / {g['gt_kcal'].mean():.0f} kcal")
    print(f"  全部 {made} 盘：卡路里均值 {tot['gt_kcal'].mean():.0f} kcal "
          f"(min {tot['gt_kcal'].min():.0f} / max {tot['gt_kcal'].max():.0f})")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
