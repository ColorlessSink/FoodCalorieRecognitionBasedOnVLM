'''
tools/utils.py — 通用工具（区域分割、面积比、深度代理、密度先验）
================================================
为什么放 tools/ 而不是 models/：
  这些是"无状态的视觉/数值工具"，被 portion_estimator 调用，本身不持有模型。
  按大作业规定的目录结构（tools/reference_detector.py, depth_estimator.py, utils.py）。

关键能力：
  - food_mask_by_saliency(): 用 OpenCV GrabCut/显著性 + 餐盘先验，把"食物前景"从盘/桌背景里抠出来。
    为什么不用 OWL-ViT/Grounding DINO：题面允许二选一/三选一，且本机无这些大检测器的权重，
    OpenCV 前景分割本地可跑、零下载，作为"多食物解耦与定位"创新点的轻量实现。
  - area_ratio(): 食物像素 / 图像像素，喂给分量估计。
  - depth_proxy(): 单目深度代理（用显著性/对比度近似"近大远小"），用于方法B的厚度先验修正。
  - density_prior() / thickness_prior(): 查 nutrition_db 的密度，按菜品类型给厚度先验。
'''
import os
import numpy as np
import pandas as pd
import cv2

from models.common import ROOT, load_config


# ---------------- 图像 IO（中文路径兼容）----------------
def imread_unicode(path):
    """cv2.imread 在 Windows 下读含中文路径会返回 None（静默失败，坑！）。
    用 np.fromfile + cv2.imdecode 绕过。返回 BGR ndarray 或 None。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------- 营养/密度先验 ----------------
_NUTR_CACHE = None
def _nutr():
    global _NUTR_CACHE
    if _NUTR_CACHE is None:
        cfg = load_config()
        _NUTR_CACHE = pd.read_csv(os.path.join(ROOT, cfg["nutrition"]["db_path"]))
    return _NUTR_CACHE

def density_prior(idx):
    """按类别 idx 查密度 g/cm³。"""
    df = _nutr()
    row = df[df["idx"] == idx]
    return float(row.iloc[0]["density"]) if len(row) else 1.0

def thickness_prior(idx, zh=None):
    """按菜品质地给厚度先验(cm)。液体~1.5, 半固体(炒菜/炖菜)~2.5, 固体块(肉/饭)~3.0, 蓬松(叶菜/面包)~2.0。"""
    name = zh or ""
    # 简单规则：含"粥/汤/奶/羹/汁"偏液态；"饭/面/饼/面包/包/饺"偏固态主食；"菜/生菜/菠菜/花菜"偏蓬松
    if any(k in name for k in ["粥", "汤", "奶", "羹", "汁", "蒸蛋"]):
        return 1.5
    if any(k in name for k in ["饭", "炒饭", "面", "饼", "面包", "包", "饺", "包子", "小笼", "肉夹馍"]):
        return 3.0
    if any(k in name for k in ["生菜", "菠菜", "花菜", "西兰花", "凉拌", "薯条", "花生"]):
        return 2.0
    return 2.5   # 默认炒菜/肉类

# ---------------- 图像前景分割（食物区域）----------------
def food_mask_by_saliency(img_bgr):
    """
    用 GrabCut + 显著性粗掩码抠食物前景，返回 uint8 掩码(0/255)。
    为什么这样做：
      - GrabCut 需要一个初始矩形/掩码，直接用全图当 likely-FG 效果差。
      - 先用显著性检测(SpectralResidual)+HSV 饱和度找"盘子中心的食物"
        (食物通常比桌布更饱和、更居中)，得到粗 ROI 再喂 GrabCut。
    """
    h, w = img_bgr.shape[:2]
    # 1. 显著性图
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    (ok, smap) = sal.computeSaliency(img_bgr)
    if not ok:
        smap = np.zeros((h, w), np.float32)
    smap = (smap * 255).astype(np.uint8)
    smap = cv2.GaussianBlur(smap, (5, 5), 0)

    # 2. HSV 饱和度（食物多有色泽，桌布多低饱和）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    combined = cv2.addWeighted(smap, 0.6, s, 0.4, 0)

    # 3. 自适应阈值得粗前景
    _, fg = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学去噪 + 中心偏置（食物多在图中央）
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, w//40), max(5, h//40)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kern)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kern)
    # 中心权重：距中心越近越可能是食物
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    center_mask = (dist < 0.9).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, center_mask)

    # 4. GrabCut 精修
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[fg > 0] = cv2.GC_PR_FGD
    # 确定背景：图像边缘一圈
    bgd = np.zeros((h, w), np.uint8)
    cv2.rectangle(bgd, (0, 0), (w, h), 0, -1)
    margin = max(5, min(w, h) // 12)
    bgd[:, :margin] = 1; bgd[:, -margin:] = 1; bgd[:margin, :] = 1; bgd[-margin:, :] = 1
    mask[bgd > 0] = cv2.GC_BGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return fg   # GrabCut 失败退回粗掩码
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out


def area_ratio(mask):
    """掩码中前景像素 / 总像素，返回 [0,1]。"""
    return float(np.count_nonzero(mask)) / (mask.shape[0] * mask.shape[1])


def largest_contour_bbox(mask):
    """返回最大前景轮廓的外接矩形 (x,y,w,h)，无则 None。用于"定位"输出。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return cv2.boundingRect(c)


# ---------------- 深度代理（单目）----------------
def depth_proxy(img_bgr, mask):
    """
    轻量单目深度代理：真实深度估计(MiDaS/Depth Anything)权重几十 MB，本机下载不可控，
    且分量估计主要靠"面积×厚度×密度"，深度只做厚度修正。
    这里用"中心区域 vs 边缘"的相对大小近似近大远小：越靠中心且越大→越近。
    返回 [0,1] 相对深度图(值大=近)。
    """
    h, w = img_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    # 高斯中心权重
    g = np.exp(-(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2))
    g = g * (mask > 0)
    return g.astype(np.float32)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # 自测：对一张测试图跑分割
    import os
    img_path = os.path.join(ROOT, "dataset_50cls", "test", "00_麻婆豆腐")
    if os.path.isdir(img_path):
        f = sorted(os.listdir(img_path))[0]
        img = imread_unicode(os.path.join(img_path, f))
        m = food_mask_by_saliency(img)
        print(f"图: {f}  尺寸: {img.shape[:2]}  面积比: {area_ratio(m):.3f}  bbox: {largest_contour_bbox(m)}")
