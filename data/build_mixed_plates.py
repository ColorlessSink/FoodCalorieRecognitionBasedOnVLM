'''
合成混合餐盘数据脚本（单食物 → 多食物同框）
---
ChineseFoodNet 每张图只有单一食物、无混合餐盘图。从已有 test 集合成：
每盘取 2~4 张不同类的 test 图（已带与单食物评估完全同口径的合成真值），
圆形羽化裁剪贴到"暖灰桌面 + 白色餐盘"画布上

布局设计（v3 随机化）：
  v1 固定三点位 → 检测走廊粘连通（召回 58%）；v2 固定布局调通（119/120）
  但 120 盘全部长一个样，检测器等于在"同分布拷贝"上评估，换一点外观就崩。
  v3 保留两条硬约束（它们是检测正确性的物理基础）：
    ① 任意两圆：圆心距 - 半径和 ≥ GAP_MIN(40px)，走廊不被羽化+std核填满
    ② 任意圆：hypot(nx,ny)+r/320 ≤ 0.90，圆整体在中心椭圆 dist<0.9 内
       （与显著性分割的中心偏置掩膜一致，边缘食物不会被裁掉）
  在约束内随机化四个维度（半径/位置/外观/组成）：
    - 直径：双菜 190~240、三菜 170~210、四菜 150~185（随机每盘抽）
    - 位置：在椭圆内随机撒点，拒绝间隙不足的候选（拒绝采样）
    - 外观：桌色（暖灰/冷灰/木色系）、盘色（白/米白/浅灰、半径 270~300）、
      亮度乘子 0.85~1.10、噪声 σ 1.5~3.0、JPEG 质量 85~95
    - 组成：2/3/4 菜混合（60/30/10%），直径与数量联动保证放得下

可复现性纪律（v3 起两条独立随机流）：
  rng_pick = random.Random(SEED) 只管组件选取，rng_look = random.Random(SEED+1)
  只管布局/外观/JPEG 质量。拆流后改布局/外观绝不影响选图，实验可单独归因

生成（seed=2026 可复现）：
  dataset_50cls/mixed/plate_XXXX.jpg        120 张（约 72 双菜 + 36 三菜 + 12 四菜）
  dataset_50cls/test_mixed.csv              每组件一行：
      plate,n_comp,comp,src_path,label,name,weight_g_true,calories_kcal_true,cx,cy,dia
'''
import os, sys, random
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import ROOT, load_config
from tools.utils import imread_unicode

SEED = 2026
N_PLATES = 120
CANVAS = 640
SRC_CROP_FRAC = 0.42      # 源图中心圆半径 = 0.42 × 短边（食物多居中）
GAP_MIN = 40               # 两圆最小间隙（检测走廊安全下界）

# v3 直径范围（按组件数联动，保证在 300 半径盘内放得下 + 间隙 ≥ GAP_MIN）
DIA_RANGE = {2: (190, 240), 3: (170, 210), 4: (150, 185)}

# 外观随机化参数（都在"白盘 + 低显著性桌底"的检测假设内）
DESK_PALETTES = [
    (150, 142, 132),   # 暖灰（v2 原色）
    (128, 130, 136),   # 冷灰
    (139, 116, 92),    # 浅木色
]
PLATE_FILL = [(238, 236, 232), (232, 228, 220), (228, 226, 229)]     # 白/米白/浅灰盘
PLATE_BORDER = [(205, 202, 196), (198, 193, 184), (201, 199, 203)]


def _ellipse_ok(cx, cy, r):
    # 约束②：圆整体在中心椭圆 dist<0.9 内（nx,ny 为归一化坐标）
    # 界上留 0.002 裕量：int(round()) 量化最多再吃 ~0.002，不留裕量会出现
    # 0.9001 这种"名义达标、实测破线"的圆
    nx = (cx - CANVAS / 2) / (CANVAS / 2)
    ny = (cy - CANVAS / 2) / (CANVAS / 2)
    return (nx * nx + ny * ny) ** 0.5 + r / (CANVAS / 2) <= 0.90 - 0.002


def sample_layout(rng, n_comp):
    # v3 随机布局：先抽每圆直径，再在椭圆内拒绝采样圆心（间隙 ≥ GAP_MIN）
    # 500 次失败兜底退化为 v2 固定布局（约束已知满足）
    # 间隙检查留 2px 裕量：圆心 int(round()) 量化最多吃掉 ~0.7px×2
    GAP_CHECK = GAP_MIN + 2
    dias = [rng.randint(*DIA_RANGE[n_comp]) for _ in range(n_comp)]
    circles = []
    for _ in range(500):
        ok = True
        for d in dias:
            r = d / 2.0
            # 拒绝采样：随机圆心需同时满足椭圆约束 + 与已放圆的间隙
            for _try in range(200):
                cx = rng.uniform(r + 20, CANVAS - r - 20)
                cy = rng.uniform(r + 20, CANVAS - r - 20)
                if not _ellipse_ok(cx, cy, r):
                    continue
                if all(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 >= r + rr + GAP_CHECK
                       for (x, y), rr in circles):
                    circles.append(((cx, cy), r))
                    break
            else:
                ok = False
                break
        if ok and len(circles) == n_comp:
            return [((int(round(c[0][0])), int(round(c[0][1]))), int(round(c[1] * 2)))
                    for c in circles]
        circles = []
    # 兜底：v2 固定布局（约束已验证）
    fallback = {
        2: [((180, 320), 230), ((460, 320), 230)],
        3: [((320, 200), 200), ((185, 400), 200), ((455, 400), 200)],
    }
    return fallback.get(n_comp, fallback[2])


def make_canvas(rng):
    # v3 外观随机化：桌色三系 × 亮度乘子 × 噪声；盘色三档 × 盘径 270~300
    # 都保持"桌面平滑 + 白盘低饱和"的检测假设（局部 std 上食物纹理仍是双峰主信号）
    base = DESK_PALETTES[rng.randrange(len(DESK_PALETTES))]
    bright = rng.uniform(0.85, 1.10)
    noise_s = rng.uniform(1.5, 3.0)
    c = np.full((CANVAS, CANVAS, 3), base, np.uint8)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    d = np.sqrt(((xx - CANVAS / 2) / (CANVAS / 2)) ** 2 +
                ((yy - CANVAS / 2) / (CANVAS / 2)) ** 2)
    shade = (1.0 - 0.10 * d).clip(0.8, 1.0)
    c = (c.astype(np.float32) * bright * shade[..., None] +
         rng.gauss(0, noise_s) * np.ones_like(c, np.float32)).clip(0, 255).astype(np.uint8)
    # 白色餐盘：随机盘色/盘径（270~300）
    pi = rng.randrange(len(PLATE_FILL))
    pr = rng.randint(270, 300)
    cv2.ellipse(c, (CANVAS // 2, CANVAS // 2), (pr, pr), 0, 0, 360, PLATE_FILL[pi], -1)
    cv2.ellipse(c, (CANVAS // 2, CANVAS // 2), (pr, pr), 0, 0, 360, PLATE_BORDER[pi], 3)
    return c, pr


def paste_circle(canvas, src_bgr, cx, cy, dia):
    # 把源图中心圆形区域（羽化边缘）贴到画布 (cx,cy)，直径 dia。成功返回 True
    h, w = src_bgr.shape[:2]
    r = int(SRC_CROP_FRAC * min(h, w))
    if r < 20:
        return False
    alpha = np.zeros((h, w), np.float32)
    cv2.circle(alpha, (w // 2, h // 2), r, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.0, r / 20.0))     # 羽化，避免硬边
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
    print(f"test 图 {len(df)} 张（带真值）")

    # 两条随机流（可复现性纪律，见文件头注释）：
    #   rng_pick —— 组件选取（拆流后具体选图与 v2 不同，只换图不换口径）
    #   rng_look —— 布局/外观/JPEG 质量（改它不影响组件选取，可单独归因）
    rng_pick = random.Random(SEED)
    rng_look = random.Random(SEED + 1)
    out_dir = os.path.join(data_dir, "mixed")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    # 组成配比：60% 双菜 / 30% 三菜 / 10% 四菜（双菜仍占多数，与 v2 可比）
    n_double = int(N_PLATES * 0.6)
    n_triple = int(N_PLATES * 0.3)
    made = 0
    for i in range(N_PLATES):
        n_comp = 2 if i < n_double else (3 if i < n_double + n_triple else 4)
        labels = rng_pick.sample(sorted(df["label"].unique().tolist()), n_comp)
        picks = []
        for lb in labels:
            sub = df[df["label"] == lb]
            picks.append(sub.iloc[rng_pick.randrange(len(sub))])

        layout = sample_layout(rng_look, n_comp)
        canvas, _pr = make_canvas(rng_look)
        jpeg_q = int(rng_look.randint(85, 95))
        pid = f"plate_{i + 1:04d}"
        ok = True
        comps = []
        for c, (r, ((cx, cy), dia)) in enumerate(zip(picks, layout)):
            src = imread_unicode(os.path.join(data_dir, r["key"]))
            if src is None:
                ok = False
                break
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

        jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])[1]
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
    print(f"\n已写入 {out_dir}/ ({made} 张)  + test_mixed.csv ({len(out)} 组件行)")
    for k, g in tot.groupby("n_comp"):
        print(f"  {k} 菜盘 {len(g)} 张：均值 {g['gt_w'].mean():.0f}g / {g['gt_kcal'].mean():.0f} kcal")
    print(f"  全部 {made} 盘：卡路里均值 {tot['gt_kcal'].mean():.0f} kcal "
          f"(min {tot['gt_kcal'].min():.0f} / max {tot['gt_kcal'].max():.0f})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
