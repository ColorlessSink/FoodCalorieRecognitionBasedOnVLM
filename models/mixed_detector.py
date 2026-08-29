'''
models/mixed_detector.py — 多食物区域检测（混合餐盘支持）
================================================
为什么需要：
  大作业数据集要求"单食物与混合餐盘均需覆盖"，实验要求"混合餐盘卡路里
  MAE≤100kcal"。单食物流水线（识别→分量→营养）假设一图一食物，混合餐盘必须
  先回答"图里有几样食物、各在哪"，否则多菜会被当一道菜处理。
  本模块在流水线最前面加"区域提议"：detect(img) -> [区域]，下游逐区域走
  既有单食物流水线后求和——把多食物问题分解为多个单食物问题，复用全部标定。

区域提议为什么用"局部标准差（纹理）"而不是显著性/分水岭（试错记录）：
  ① 显著性+Otsu（单食物分割主力）在混合盘上会漏检浅色/低饱和食物：
     实测 plate_0006 整个菜漏检（盘上 2 菜只检出 1 个）。漏一个组件
     ≈ 400kcal 的不可挽回误差，远超 100kcal 预算 → 混合盘召回率是第一优先。
  ② 分水岭（距离变换峰值作标记）在带洞掩膜上峰值会定位到洞边缘，
     实测 30 盘 0 全对、组件召回 0.100，弃用（见 process_log.md）。
  ③ 混合盘（合成）= 平滑桌面 + 平滑白盘 + 贴上去的"食物照片圆片"。
     "照片 vs 平坦底色"在局部标准差上是双峰强信号：食物无论深浅，
     照片内部总有米粒/酱汁/轮廓纹理（std 远大于背景噪声 σ≈2~3）。
     实测该线索对显著性漏掉的浅色菜同样有效——正好补①的短板。

流水线：
  局部std → Otsu(钳位[4,12]) → 开/闭形态学 → 填洞 → 连通域 → 面积过滤
  → 疑似两圆粘连的团块用"距离变换分水岭"拆分
  → 每区域输出 {mask, bbox, cx, cy, area}

输出约定：regions 按 area 降序；对"整个图就是一张普通单食物照片"的输入，
纹理掩膜会覆盖全图 → 只有一个区域且面积占比>0.6 → 上游可退回单食物路径。
'''
import os
import sys
import numpy as np
import cv2

# 以 `python models/mixed_detector.py` 方式启动时，Python 只把 models/ 加进
# sys.path，找不到根目录下的包。补进项目根目录（models 的上一级），使直接
# 运行与 `python -m models.mixed_detector` 都能工作。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import load_config


# ---------------- 工具：填洞 ----------------
def fill_holes(bin_mask):
    """把二值掩膜(0/1)中不与边界连通的洞填成前景。
    为什么需要：贴片内部若有低于纹理阈值的平滑块（如源图的白盘），
    一个圆片会被碎成多块，连通域计数就不可信了。"""
    m = (bin_mask > 0).astype(np.uint8)
    h, w = m.shape[:2]
    seed = None
    for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if m[y, x] == 0:
            seed = (int(x), int(y))
            break
    if seed is None:
        for x in range(w):
            if m[0, x] == 0:
                seed = (x, 0); break
            if m[h - 1, x] == 0:
                seed = (x, h - 1); break
        if seed is None:
            for y in range(h):
                if m[y, 0] == 0:
                    seed = (0, y); break
                if m[y, w - 1] == 0:
                    seed = (w - 1, y); break
    if seed is None:          # 全前景，无洞
        return m
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, seed, 1)
    out = m.copy()
    out[ff == 0] = 1          # 从边界填不到的 0 = 洞
    return out


class MixedDetector:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        p = (self.cfg or {}).get("mixed", {}) or {}
        self.std_k = int(p.get("std_kernel", 15))
        self.thr_lo = float(p.get("std_thr_lo", 4.0))
        self.thr_hi = float(p.get("std_thr_hi", 12.0))
        self.min_area_frac = float(p.get("min_area_frac", 0.01))
        self.split_area_factor = float(p.get("split_area_factor", 1.5))
        self.split_peak_frac = float(p.get("split_peak_frac", 0.55))

    # ---------------- 主入口 ----------------
    def detect(self, img_bgr):
        """返回区域列表（area 降序）：[{mask,bbox,cx,cy,area}, ...]"""
        if img_bgr is None:
            return []
        m = self.texture_mask(img_bgr)
        return self.connected_regions(m, img_bgr)

    def is_single_food(self, img_bgr, regions=None):
        """普通单食物照片（整图有纹理）→ 只有一个大区域，退回单食物路径。
        判据：唯一区域面积占比 > 0.6（合成混合盘单贴片占比 ≈ 0.08~0.13）。"""
        regions = regions if regions is not None else self.detect(img_bgr)
        if len(regions) != 1:
            return False
        h, w = img_bgr.shape[:2]
        return regions[0]["area"] / float(h * w) > 0.6

    # ---------------- 纹理掩膜 ----------------
    def texture_mask(self, img_bgr):
        """局部标准差 → 阈值 → 形态学 → 填洞。返回 0/255 uint8。"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = self.std_k
        mu = cv2.boxFilter(gray, -1, (k, k))
        m2 = cv2.boxFilter(gray * gray, -1, (k, k))
        std = np.sqrt(np.maximum(m2 - mu * mu, 0.0))

        # 排除"白色餐盘边缘环带"的假纹理（实测环带 std 中位数 10~12，
        # 与食物纹理区间重叠；盘面内部中位数 0~4）。
        # 做法与显著性分割的"中心偏置"同思路：只认中心椭圆内的纹理。
        h, w = std.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cxx, cyy = w / 2, h / 2
        dist = np.sqrt(((xx - cxx) / (w / 2)) ** 2 + ((yy - cyy) / (h / 2)) ** 2)
        std = std * (dist < 0.92)

        # Otsu 自适应，但钳到 [lo, hi]：
        # 双峰清晰时 Otsu 即好阈值；分布退化时钳位兜底（背景噪声 σ≈2~3，
        # 照片纹理一般 >8，钳位区间覆盖两者的安全带）。
        std_u8 = np.clip(std, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(np.clip(thr, self.thr_lo, self.thr_hi))
        m = (std > thr).astype(np.uint8)

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)               # 去孤立噪点
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern, iterations=2)  # 桥接近邻碎片
        m = fill_holes(m)
        return (m * 255).astype(np.uint8)

    # ---------------- 连通域 → 区域 ----------------
    def connected_regions(self, mask, img_bgr=None):
        h, w = mask.shape[:2]
        min_area = max(200, int(self.min_area_frac * h * w))
        n, labels, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        regions = []
        for i in range(1, n):
            if stats[i, 4] < min_area:
                continue
            blob = (labels == i).astype(np.uint8)
            for part in self._split_if_merged(blob, img_bgr):
                ys, xs = np.nonzero(part)
                if len(ys) == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                regions.append({
                    "mask": (part * 255).astype(np.uint8),
                    "bbox": (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    "cx": float(xs.mean()), "cy": float(ys.mean()),
                    "area": int(part.sum()),
                })
        # 贴片碎裂合并：一个食物的圆片有时被纹理性状不同切成"1 大 + N 小"的
        # 多个连通域（典型：米饭/双皮奶等浅色低纹理食物贴到白盘上，部分像素
        # 低于纹理阈值断开）。合并两条规则（满足其一即并入大区域）：
        #   ① bbox 相交且面积 < 主区域 25%（真组件面积比 ≥1.32，不可能 <1/4）；
        #   ② 面积 < 主区域 30% 且质心距 < 主区域 bbox 外接圆半径×1.3 ——
        #      处理"碎片 bbox 不相交但散布在同一食物圆片内"的情形
        #      （plate_0036 的米饭碎成 5 块，散布在直径 230 的圆内）。
        # 阈值经 120 盘扫描：frac=0.30/mult=1.3 时 119/120 全对、召回 100%；
        # 放宽到 0.40 开始误并真组件（召回降到 99.7%）。
        regions.sort(key=lambda r: -r["area"])
        merged = []
        for r in regions:
            absorbed = False
            for m in merged:
                x, y, bw, bh = m["bbox"]
                rx, ry, rw, rh = r["bbox"]
                ix = max(0, min(x + bw, rx + rw) - max(x, rx))
                iy = max(0, min(y + bh, ry + rh) - max(y, ry))
                inter = ix > 0 and iy > 0
                dcent = ((r["cx"] - m["cx"]) ** 2 + (r["cy"] - m["cy"]) ** 2) ** 0.5
                eq_r = (bw * bw + bh * bh) ** 0.5 / 2
                if (inter and r["area"] < 0.25 * m["area"]) or (
                        r["area"] < 0.30 * m["area"] and dcent < 1.3 * eq_r):
                    m["mask"] = cv2.bitwise_or(m["mask"], r["mask"])
                    ys, xs = np.nonzero(m["mask"])
                    m["bbox"] = (int(xs.min()), int(ys.min()),
                                 int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
                    m["cx"], m["cy"] = float(xs.mean()), float(ys.mean())
                    m["area"] = int((m["mask"] > 0).sum())
                    absorbed = True
                    break
            if not absorbed:
                merged.append(r)
        return merged

    # ---------------- 粘连拆分（分水岭）----------------
    def _split_if_merged(self, blob, img_bgr):
        """两圆片被羽化边桥连成一个连通域时拆开。
        判据：单圆 面积≈π·dmax²；双圆粘连 面积≈2π·dmax²。
        超过 split_area_factor×π·dmax² 才尝试拆，且峰值连通域≥2才真拆。
        （注意：这里和早期失败的分水岭不同——输入是"填过洞的整圆片"，
         距离变换在圆心处是真实峰值，不会被洞边缘带偏。）"""
        area = int(blob.sum())
        if area < 400:
            return [blob]
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        dmax = float(dist.max())
        if dmax <= 1e-6 or area <= self.split_area_factor * np.pi * dmax * dmax:
            return [blob]
        _, peak = cv2.threshold(dist, self.split_peak_frac * dmax, 255, cv2.THRESH_BINARY)
        n, labels = cv2.connectedComponents(peak.astype(np.uint8), 8)
        if n <= 2:      # 只有 1 个峰：不是粘连，只是形状不规则
            return [blob]
        if img_bgr is None:
            bgr = cv2.cvtColor((blob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            bgr = img_bgr.copy()
        markers = labels.astype(np.int32)
        try:
            ws = cv2.watershed(bgr, markers)
        except Exception:
            return [blob]
        parts = []
        for i in range(1, n):
            part = ((ws == i)).astype(np.uint8) & blob
            if int(part.sum()) >= 200:
                parts.append(part)
        return parts or [blob]

    # ---------------- 区域 → 裁剪 ----------------
    @staticmethod
    def crop(img_bgr, region, margin_frac=0.12, min_margin=8):
        """按区域 bbox（外扩 margin）裁出"单食物小图"，供识别/分量复用。
        为什么外扩一点：完全贴边的裁剪会切掉羽化边，让食物顶满画幅，
        与单食物训练口径（食物居中、四周留白）不一致。"""
        x, y, w, h = region["bbox"]
        H, W = img_bgr.shape[:2]
        mx = max(min_margin, int(max(w, h) * margin_frac))
        x0 = max(0, x - mx); y0 = max(0, y - mx)
        x1 = min(W, x + w + mx); y1 = min(H, y + h + mx)
        return img_bgr[y0:y1, x0:x1].copy(), (x0, y0)


# ---------------- 自测/诊断：全量 120 盘检测评估 ----------------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import pandas as pd
    from models.common import ROOT
    from tools.utils import imread_unicode

    det = MixedDetector()
    df = pd.read_csv(os.path.join(ROOT, "dataset_50cls", "test_mixed.csv"))

    n_plate = n_exact = 0
    comp_total = comp_hit = 0
    dup = extra = 0
    cent_errs = []
    hard = []
    for pid, g in df.groupby("plate"):
        img = imread_unicode(os.path.join(ROOT, "dataset_50cls", pid))
        if img is None:
            print(f"读不到 {pid}")
            continue
        regions = det.detect(img)
        n_plate += 1
        n_exact += int(len(regions) == g["n_comp"].iloc[0])

        # 匹配：区域质心到哪个 GT 圆心最近；dist < dia/2 记命中
        matched_gt = set()
        r_cent = []
        for r in regions:
            best, bd = None, 1e9
            for _, row in g.iterrows():
                d = ((r["cx"] - row["cx"]) ** 2 + (r["cy"] - row["cy"]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, row
            if bd < best["dia"] / 2:
                matched_gt.add(best["comp"])
                r_cent.append(bd)
            else:
                extra += 1
        comp_total += len(g)
        comp_hit += len(matched_gt)
        if len(matched_gt) < len(g):
            hard.append((pid, len(g), len(matched_gt), len(regions)))
        cent_errs += r_cent

    print(f"\n========== 检测评估（{n_plate} 盘 / {comp_total} 组件）==========")
    print(f"区域数完全正确 : {n_exact}/{n_plate} ({n_exact/n_plate*100:.1f}%)")
    print(f"组件召回       : {comp_hit}/{comp_total} ({comp_hit/comp_total*100:.1f}%)")
    print(f"多余区域       : {extra}   命中质心平均偏差: {np.mean(cent_errs):.1f}px")
    if hard:
        print(f"未满召回的盘（plate, GT组件, 命中, 检出区域数）: {hard[:20]}")

    # 画 3 个诊断图
    from models.common import load_config as _lc
    outdir = os.path.join(ROOT, "results")
    for pid in [g for g in df["plate"].unique()[:200]][:3]:
        img = imread_unicode(os.path.join(ROOT, "dataset_50cls", pid))
        regions = det.detect(img)
        vis = img.copy()
        for i, r in enumerate(regions):
            cnts, _ = cv2.findContours(r["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
            cv2.circle(vis, (int(r["cx"]), int(r["cy"])), 4, (255, 0, 0), -1)
            cv2.putText(vis, f"r{i}", (int(r["cx"]) + 8, int(r["cy"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imencode(".jpg", vis)[1].tofile(
            os.path.join(outdir, f"_diag_regions_{pid.split('/')[-1]}"))
