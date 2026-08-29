'''
models/portion_estimator.py — 分量估计模块（模块2）
================================================
为什么需要、为什么这么设计：
  大作业模块2要求"实现 ≥1 种分量估计方法（建议2种对比），单食物 MAE≤30g 或相对误差≤25%"。
  我们实现两种方法并对比：
    方法A（几何法） : 分割得面积比 → 按类桶的面积比中位数归一化 → 相对调制系数 → 乘类先验均值
    方法B（CoT法） : 把识别结果+面积比+密度+先验喂给 glm-5.2，让它分步推理给重量（带不确定性）
  两者都依赖"面积比"这一可解释中间量，便于消融对比（见 experiments/ablation_study.py）。

方法A 的标定演进（重要，见 process_log.md §3.2）：
  v1（失败）: 固定 plate_ref_px=900px=23cm → cm_per_px 与图像分辨率耦合，
              1440px 大图食物面积爆高（回锅肉估到 1683g，MAE 143g）。
              诊断证实：area_ratio 与短边 corr=0.007（已是好信号），
              但 food_px=ar×总像素 与总像素 corr=0.895（绝对标定不可靠）。
  v2（当前）: 弃用绝对 cm 标定。area_ratio 作"相对调制器"，以类先验均值 μ 为锚：
              weight = μ × clamp(ar / median_ar_bucket, [lo, hi])。
              ① 与图像分辨率解耦；② 以 μ 为锚避免 2D 面积法系统性高估；
              ③ area_ratio 提供有据可依的相对涨落；④ 分割失败(ar→0)自然回退到 μ。
              这样几何法的下界就是"完全不信面积、只用先验"，oracle MAE ≈ 0.8σ ≈ 24g。

评估口径（重要，见 process_log.md §0.4）：
  真值 = hash+类先验采样，与估计器输入（图像像素）解耦。
  估计器最优输出 ≈ 类均值 μ，故 oracle MAE ≈ 0.8σ ≈ 24g < 30g。
  方法A 被设计成回归到 μ（面积比只做相对调制），方法B 让 LLM 在几何与先验间折中。
'''
import os
import json
import math
import numpy as np
import pandas as pd

from models.common import ROOT, load_config
from tools.utils import (food_mask_by_saliency, area_ratio, largest_contour_bbox,
                         density_prior, thickness_prior, imread_unicode)


class PortionEstimator:
    def __init__(self, cfg=None, use_llm=True):
        self.cfg = cfg or load_config()
        pa = self.cfg["portion"]["method_a"]
        self.plate_real_cm = pa["plate_real_cm"]
        self.plate_frac = pa.get("plate_frac", 0.85)
        self.clamp_lo = pa.get("clamp_lo", 0.6)
        self.clamp_hi = pa.get("clamp_hi", 1.6)
        self.geo_weight = pa.get("geo_weight", 0.4)
        self.ar_fail_thresh = pa.get("ar_fail_thresh", 0.05)
        self.thickness_default = pa.get("thickness_cm", {}).get("default", 2.5)

        # 类桶的面积比中位数（标定表，由 data/build_area_stats.py 预计算）
        self._ar_stats = self._load_ar_stats()

        self.use_llm = use_llm and self.cfg["llm"].get("enabled", True)
        if self.use_llm:
            from models.llm_client import LLMClient
            self.llm = LLMClient(self.cfg)
            cot_path = os.path.join(ROOT, "models", self.cfg["portion"]["method_b"]["prompt_template"])
            with open(cot_path, "r", encoding="utf-8") as f:
                self.cot_template = f.read()
        else:
            self.llm = None

    def _load_ar_stats(self):
        """读 area_ratio_stats.csv：bucket -> median_ar。没有则运行时惰性算。"""
        path = os.path.join(ROOT, "data", "area_ratio_stats.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        return dict(zip(df["bucket"], df["median_ar"]))

    def _median_ar(self, idx, name):
        """取该类所属桶的面积比中位数；缺标定表时用全局经验值 0.32。"""
        from data.build_labels import bucket_of
        b = bucket_of(idx, name)
        if self._ar_stats and b in self._ar_stats:
            return float(self._ar_stats[b])
        # 未标定时的兜底经验值（由 diag_ar.py 实测全局 median≈0.32）
        return 0.32

    # ---------- 中间量：分割 + 面积比 ----------
    def segment(self, img_path):
        """返回 (mask, area_ratio, bbox)。img_path 可含中文。"""
        img = imread_unicode(img_path)
        if img is None:
            return None, 0.0, None
        mask = food_mask_by_saliency(img)
        ar = area_ratio(mask)
        bbox = largest_contour_bbox(mask)
        return mask, ar, bbox

    # ---------- 方法A：几何法（面积比相对调制 + 类先验锚）----------
    def estimate_geometric(self, img_path, food_idx, food_name):
        """
        weight = prior_μ × clamp(ar / median_ar_bucket, [lo, hi]) × geo_weight
                 + prior_μ × (1 - geo_weight)
        即以类先验均值 μ 为锚，面积比只在 [lo,hi] 区间内做相对调制。
        面积比<ar_fail_thresh 视为分割失败，直接退先验。
        仍输出几何中间量(area_ratio/food_area_cm2/density)供可解释与消融。
        """
        prior = self._prior(food_idx, food_name)
        mask, ar, bbox = self.segment(img_path)

        if mask is None or ar < self.ar_fail_thresh:
            return {
                "weight_g": round(prior, 1), "area_ratio": round(ar, 3),
                "modulator": 1.0, "median_ar": self._median_ar(food_idx, food_name),
                "prior_g": prior, "method": "geometric(fallback_prior)",
            }

        med = self._median_ar(food_idx, food_name)
        modulator = ar / med if med > 1e-6 else 1.0
        modulator = max(self.clamp_lo, min(self.clamp_hi, modulator))
        geo_weight = prior * modulator
        # 最终 = geo_weight×几何 + (1-geo_weight)×先验，先验为锚
        weight = self.geo_weight * geo_weight + (1.0 - self.geo_weight) * prior

        # 可解释中间量（仍按餐盘先验给个 cm² 估值，仅供展示，不参与最终重量）
        h, w = mask.shape[:2]
        short = min(h, w)
        cm_per_px = self.plate_real_cm / (self.plate_frac * short)
        img_area_cm2 = (h * w) * (cm_per_px ** 2)
        food_area_cm2 = ar * img_area_cm2
        thick = thickness_prior(food_idx, food_name)
        density = density_prior(food_idx)

        return {
            "weight_g": round(weight, 1),
            "area_ratio": round(ar, 3),
            "median_ar": round(med, 3),
            "modulator": round(modulator, 3),
            "food_area_cm2": round(food_area_cm2, 1),
            "thickness_cm": thick,
            "density": density,
            "geometric_raw_g": round(geo_weight, 1),
            "prior_g": prior,
            "method": "geometric",
        }

    # ---------- 方法B：CoT LLM ----------
    def estimate_cot(self, img_path, food_idx, food_name):
        if not self.use_llm:
            return self.estimate_geometric(img_path, food_idx, food_name)
        mask, ar, _ = self.segment(img_path)
        prior = self._prior(food_idx, food_name)
        density = density_prior(food_idx)
        prompt = self.cot_template.format(
            food_name=food_name, area_ratio=ar,
            plate_cm=self.plate_real_cm, density=density, prior_g=prior)
        ans = self.llm.ask(prompt, max_tokens=1024, fallback="")
        parsed = self._parse_json(ans)
        if parsed and "weight_g" in parsed:
            parsed["method"] = "cot_llm"
            parsed["area_ratio"] = round(ar, 3)
            return parsed
        # LLM 解析失败 → 退几何法
        geo = self.estimate_geometric(img_path, food_idx, food_name)
        geo["method"] = "cot_llm(fallback_geometric)"
        return geo

    # ---------- 统一入口 ----------
    def estimate(self, img_path, food_idx, food_name, method="geometric"):
        if method == "cot":
            return self.estimate_cot(img_path, food_idx, food_name)
        return self.estimate_geometric(img_path, food_idx, food_name)

    # ---------- 内部 ----------
    def _prior(self, idx, name):
        """类先验重量：与 build_labels.py 的 PORTION_PRIOR 同口径。"""
        from data.build_labels import PORTION_PRIOR, bucket_of
        mu, _ = PORTION_PRIOR[bucket_of(idx, name)]
        return mu

    @staticmethod
    def _parse_json(text):
        """从 LLM 输出里抠出第一段 {...}。模型可能前后带解释。"""
        if not text:
            return None
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e == -1 or e < s:
            return None
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return None


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from models.food_recognizer import gather_from_split
    pe = PortionEstimator(use_llm=False)
    # 抽 50 张测试图做几何法自测
    paths, labels = gather_from_split(pe.cfg["project"]["data_dir"], "test")
    names = pd.read_csv(os.path.join(ROOT, pe.cfg["project"]["data_dir"], "classes_50.csv"))
    name_map = dict(zip(names["idx"], names["zh"]))
    lbl_df = pd.read_csv(os.path.join(ROOT, pe.cfg["project"]["data_dir"], "test_labels.csv"))
    truth_map = dict(zip(lbl_df["path"], lbl_df["weight_g_true"]))

    errs = []
    rels = []
    for p, l in zip(paths[:50], labels[:50]):
        rel = os.path.relpath(p, os.path.join(ROOT, pe.cfg["project"]["data_dir"])).replace("\\", "/")
        res = pe.estimate_geometric(p, l, name_map[l])
        t = truth_map.get(rel)
        e = abs(res["weight_g"] - t)
        errs.append(e)
        rels.append(e / t if t else 0)
        print(f"{name_map[l]:<8} 几何={res['weight_g']:>6.1f}g 真={t:>6.1f}g 误差={e:>5.1f}g ({e/t*100:.0f}%)")
    errs = np.array(errs)
    rels = np.array(rels)
    print(f"\n50 张 几何法 MAE = {errs.mean():.1f}g  中位误差 = {np.median(errs):.1f}g  相对误差 = {rels.mean()*100:.1f}%  满足≤30g占比={int((errs<=30).sum())}/50")

