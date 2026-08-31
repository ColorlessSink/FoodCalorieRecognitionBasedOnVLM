'''
SAM 混合餐盘区域检测（后端可选，接口与 MixedDetector 完全对齐）
---
背景：MixedDetector 用"局部 std 纹理 + 盘边环带 + 连通域"在合成盘上做到
119/120，但在 40 张真实混合菜网图上 37/40 整图塌成一块（域差，见
process_log §3.9）：真实菜堆叠无平坦间隔，纹理连通域天然全盘连通。
这不是调参能救的——"把一堆堆叠的菜切成 N 个语义整体"本质是实例分割
问题，传统 CV 的假设（孤立圆片 + 平坦背景）在真实域不成立。

方案：Meta SAM（ViT-B，375MB，promptable/Automatic mask generator）做
区域提议，替换纹理后端；本模块负责把 SAM 的细粒度 mask 聚合成
"一道菜"级的区域，输出 {mask,bbox,cx,cy,area}，下游 label_mixed /
mixed_eval 不用改一行。

聚合为什么这样设计（碎片 → 菜）：
  SamAutomaticMaskGenerator 会在一盘菜上吐几十~上百个 mask（每块肉/每片
  菜叶一个），它只有边界没有语义。要并回"一道菜"级区域靠四步（在
  120 盘合成 GT 上扫参验证，见 _sweep_v2/v3 脚本与本文件 __main__ 评估）：
  ① 面积/整图层过滤：min_area_frac 滤碎屑；横竖都 ≥99% 全图的整图层丢；
  ② bbox 稠密度过滤：mask 面积 / bbox 面积 < fill_thresh 的丢弃——
     实测多余区域几乎全是横带/角块/跨双菜 union（bbox 大而 mask 稀，
     fill 15~45%），菜级 mask 在自己 bbox 里致密（>70%）；
  ③ IoA-NMS：按 predicted_iou 降序保留，与已保留区域重叠 / 两者较小
     面积 ≥ ioa_thresh 就丢——同族碎片只留质量最高的那个；
  ④ 质心合并：同一道菜被拆成 2~3 块时，两碎片质心距 ≈ 两个等效半径之
     和，而相邻两道菜质心距 ≥ 半径和 + 间隙——d(质心) < beta*(r_i+r_j)
     就并（贪心，最像的先并，链式碎片也能并成一块）。
  不做"颜色/CLIP 聚类"的原因：相邻菜色接近（红烧肉+糖醋排骨），聚类
  阈值跨图不稳，风险大于收益。

约束说明：SAM 是"分割大模型"不是"视觉 LLM"——项目约束是不用 LLM 视觉
能力（glm 视觉实测不可用），SAM 只做掩码提议、零样本、本地推理，不与
约束冲突；这一点在 final_report 局限性里如实说明。

用法（与 MixedDetector 相同）：
  from models.sam_detector import SamDetector
  det = SamDetector(cfg)          # cfg["sam"] 可调参数，缺省内置
  regions = det.detect(img_bgr)   # [{mask,bbox,cx,cy,area}] area 降序
  crop, origin = SamDetector.crop(img, region)   # 静态方法直接复用父类
'''
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mixed_detector import MixedDetector


class SamDetector(MixedDetector):
    # 继承 MixedDetector 只为复用静态 crop() 与接口签名，detect() 整个
    # 被覆盖——纹理后端的所有参数（std_kernel/ring_* 等）在此不生效。

    def __init__(self, cfg=None, ckpt=None, device=None):
        self.cfg = cfg or load_config_safe()
        p = (self.cfg or {}).get("sam", {}) or {}
        # ---- SAM 本体 ----
        self.ckpt = ckpt or p.get("ckpt", "models/sam_vit_b_01ec64.pth")
        self.device = device or p.get("device", "cuda")
        # ---- MaskGenerator 可调 ----
        # points_per_side：自动提示点阵密度。16 是 SAM 论文默认；32 网图
        # 边界更细但 mask 碎片化更重、耗时×3，16 已够"菜"级边界。
        self.points_per_side = int(p.get("points_per_side", 16))
        # pred_iou_thresh：预测 IoU 低于此的 mask 直接丢（模型自评质量）。
        # 默认 0.88 只留高置信碎片——但实测"整道菜"级大 mask 的 iou 只有
        # 0.83 左右（0.6 放宽后 dish-half 0.216/0.187 才浮出来），必须降到 0.6。
        self.pred_iou_thresh = float(p.get("pred_iou_thresh", 0.6))
        # stability_score_thresh：掩码稳定性（同一提示多阈值抖动小才保留）。
        # SAM 默认 0.95 会把菜级大 mask 全杀掉（实测它们的 stab 只有 0.02~0.55，
        # 语义"一道菜"边界天然比"一个物体"抖）——必须放到 0.0，靠聚合去噪。
        self.stability_score_thresh = float(p.get("stability_score_thresh", 0.0))
        # box_nms_thresh：box-NMS 去重叠 box 的 IoU 门槛。
        self.box_nms_thresh = float(p.get("box_nms_thresh", 0.7))
        # min_mask_region_area：SAM 内部后处理直接删掉的小连通域（像素数）。
        self.min_mask_region_area = int(p.get("min_mask_region_area", 300))
        # ---- 聚合（碎片 → 菜）----
        # min_area_frac：候选面积 / 全图 < 此值当碎屑丢弃（酱汁点/葱花）。
        # 合成盘贴片占 0.08~0.13，真实菜大约 0.05~0.5；0.01 挡纯噪声。
        self.min_area_frac = float(p.get("min_area_frac", 0.01))
        # fill_thresh：mask 面积 / bbox 面积 < 此值的候选丢弃（稀疏 band/
        # 角块/跨双菜 union 的 fill 只有 15~45%，菜级 mask >70%）。
        # 调参结论：0.65 是合成盘 GT 上 exact/recall 的拐点。
        self.fill_thresh = float(p.get("fill_thresh", 0.65))
        # ioa_thresh：IoA-NMS 门槛——与已保留区域重叠 / 两者较小面积
        # ≥ 此值就丢（碎片∩整菜/碎片 ≈ 1.0，邻菜只共享细边界 <10%）。
        self.ioa_thresh = float(p.get("ioa_thresh", 0.55))
        # merge_beta：质心合并门槛系数，d(质心) < beta*(r_i+r_j) 就并
        # （r = sqrt(area/π) 等效半径）。1.15 在合成 GT 上最优：
        # beta≥1.3 会把相邻两菜焊死（recall 掉到 204/300）。
        self.merge_beta = float(p.get("merge_beta", 1.15))

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        model_type = "vit_b" if "vit_b" in os.path.basename(self.ckpt) else "vit_h"
        sam = sam_model_registry[model_type](checkpoint=self.ckpt)
        sam.to(device=self.device)
        self.sam = sam
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            box_nms_thresh=self.box_nms_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def detect(self, img_bgr):
        # 返回区域列表（area 降序）：[{mask,bbox,cx,cy,area}, ...]
        if img_bgr is None:
            return []
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        r = self.mask_generator.generate(rgb)
        regions = self._aggregate(r, img_bgr.shape[:2])
        return regions

    # ---------------- 碎片聚合：SAM masks → 菜级区域 ----------------
    def _aggregate(self, masks_data, shape_hw):
        H, W = shape_hw
        img_area = float(H * W)
        min_area = max(200, int(self.min_area_frac * img_area))

        # ①② 面积 / 整图层 / bbox 稠密度过滤
        cands = []
        for m in masks_data:
            a = int(m["area"])
            if a < min_area:
                continue
            x, y, w, h = m["bbox"]
            if w >= 0.99 * W and h >= 0.99 * H:
                continue                          # 整图背景层
            if a / (w * h) < self.fill_thresh:
                continue                          # 稀疏 band / 角块 / 跨菜 union
            cands.append(m)
        if not cands:
            return []

        # ③ IoA-NMS：按 predicted_iou 降序保留，碎片被高质量大块压掉
        cands.sort(key=lambda m: -m["predicted_iou"])
        kept = []
        for m in cands:
            seg = m["segmentation"].astype(bool)
            drop = False
            for k in kept:
                inter = int(np.logical_and(seg, k).sum())
                if inter and inter / min(int(seg.sum()), int(k.sum())) >= self.ioa_thresh:
                    drop = True
                    break
            if not drop:
                kept.append(seg)
        if not kept:
            return []

        # ④ 质心合并：同菜碎片质心距 ≈ 等效半径之和，贪心最像的先并
        def _stats(mask):
            ys, xs = np.nonzero(mask)
            return float(xs.mean()), float(ys.mean()), (int(mask.sum()) / np.pi) ** 0.5

        while len(kept) > 1:
            best = None
            st = [_stats(m) for m in kept]
            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    d = ((st[i][0] - st[j][0]) ** 2 + (st[i][1] - st[j][1]) ** 2) ** 0.5
                    slack = self.merge_beta * (st[i][2] + st[j][2]) - d
                    if slack > 0 and (best is None or slack > best[0]):
                        best = (slack, i, j)
            if best is None:
                break
            _, i, j = best
            kept[i] = kept[i] | kept.pop(j)

        # 生成 regions（mask/bbox/cx/cy/area），按 area 降序
        regions = []
        for seg in kept:
            ys, xs = np.nonzero(seg)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            regions.append({
                "mask": (seg.astype(np.uint8) * 255),
                "bbox": (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                "cx": float(xs.mean()), "cy": float(ys.mean()),
                "area": int(seg.sum()),
            })
        regions.sort(key=lambda r: -r["area"])
        return regions


def load_config_safe():
    # 顶层 `python models/sam_detector.py` 自测时 sys.path 未含项目根
    try:
        from models.common import load_config
        return load_config()
    except Exception:
        return None


# ---------------- 自测/诊断：合成盘 GT 评估 + 真实图落盘 ----------------
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import pandas as pd
    from models.common import ROOT
    from tools.utils import imread_unicode

    det = SamDetector()
    df = pd.read_csv(os.path.join(ROOT, "dataset_50cls", "test_mixed.csv"))
    outdir = os.path.join(ROOT, "results")

    n_plate = n_exact = 0
    comp_total = comp_hit = extra = 0
    cent_errs, hard = [], []
    for pid, g in df.groupby("plate"):
        img = imread_unicode(os.path.join(ROOT, "dataset_50cls", pid))
        if img is None:
            print(f"读不到 {pid}")
            continue
        regions = det.detect(img)
        n_plate += 1
        n_exact += int(len(regions) == g["n_comp"].iloc[0])
        matched_gt = set()
        for r in regions:
            best, bd = None, 1e9
            for _, row in g.iterrows():
                d = ((r["cx"] - row["cx"]) ** 2 + (r["cy"] - row["cy"]) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, row
            if bd < best["dia"] / 2:
                matched_gt.add(best["comp"])
                cent_errs.append(bd)
            else:
                extra += 1
        comp_total += len(g)
        comp_hit += len(matched_gt)
        if len(matched_gt) < len(g) or len(regions) != g["n_comp"].iloc[0]:
            hard.append((pid, len(g), len(matched_gt), len(regions)))

    print(f"\n========== SAM 检测评估（{n_plate} 盘 / {comp_total} 组件）==========")
    print(f"区域数完全正确 : {n_exact}/{n_plate} ({n_exact/n_plate*100:.1f}%)")
    print(f"组件召回       : {comp_hit}/{comp_total} ({comp_hit/comp_total*100:.1f}%)")
    print(f"多余区域       : {extra}   命中质心平均偏差: {np.mean(cent_errs):.1f}px")
    if hard:
        print(f"问题盘 (plate, GT组件, 命中, 检出): {hard[:25]}")

    # 诊断图：挑问题最多的前 3 盘落盘
    vis_ids = [h[0] for h in hard[:3]] or df["plate"].unique()[:3].tolist()
    for pid in vis_ids:
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
            os.path.join(outdir, f"_diag_sam_{pid.split('/')[-1]}"))
    print(f"诊断图 -> {outdir}/_diag_sam_*")
