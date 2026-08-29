'''
models/nutrition_querier.py — 营养查询与卡路里计算模块（模块3）
================================================
为什么需要：
  模块2 给出"食物类别 + 重量(g)"，模块3 负责：
    ① 查营养库得每100g的 kcal/蛋白/脂肪/碳水；
    ② 按估计重量换算成单份总热量与三大宏量；
    ③ 输出结构化 JSON（喂给智能体模块做多轮对话解释）。

设计：
  - NutritionQuerier 一次读 nutrition_db.csv 缓存，按 idx 查。
  - compute(food_idx, weight_g) 返回 {food_name, weight_g, kcal, protein_g, fat_g, carbs_g, per100}。
  - 默认取 module2 的 PortionEstimator 结果作为 weight_g，也可外部传入（便于"识别+分量+营养"端到端）。
  - 多食物：compute_meal(list_of_(idx,weight)) 汇总一餐。
'''
import os
import json
import pandas as pd

from models.common import ROOT, load_config


class NutritionQuerier:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        path = os.path.join(ROOT, self.cfg["nutrition"]["db_path"])
        self.db = pd.read_csv(path)
        self._by_idx = {int(r["idx"]): r for _, r in self.db.iterrows()}

    def query(self, food_idx):
        """返回该类每 100g 的营养字典；找不到返回 None。"""
        r = self._by_idx.get(int(food_idx))
        if r is None:
            return None
        return {
            "idx": int(r["idx"]),
            "zh": r["zh"], "en": r["en"],
            "kcal_per100g": float(r["kcal_per100g"]),
            "protein_g": float(r["protein_g"]),
            "fat_g": float(r["fat_g"]),
            "carbs_g": float(r["carbs_g"]),
            "density": float(r["density"]),
        }

    def compute(self, food_idx, weight_g):
        """单食物：按重量换算总热量与宏量。"""
        q = self.query(food_idx)
        if q is None:
            return {"food_idx": int(food_idx), "weight_g": float(weight_g),
                    "kcal": 0.0, "error": "未在营养库找到该类别"}
        scale = float(weight_g) / 100.0
        return {
            "food_idx": int(food_idx),
            "food_name": q["zh"],
            "weight_g": round(float(weight_g), 1),
            "kcal": round(q["kcal_per100g"] * scale, 1),
            "protein_g": round(q["protein_g"] * scale, 1),
            "fat_g": round(q["fat_g"] * scale, 1),
            "carbs_g": round(q["carbs_g"] * scale, 1),
            "per100": {
                "kcal": q["kcal_per100g"], "protein_g": q["protein_g"],
                "fat_g": q["fat_g"], "carbs_g": q["carbs_g"], "density": q["density"],
            },
        }

    def compute_meal(self, items):
        """一餐多食物汇总。items: [(food_idx, weight_g), ...]。返回每项 + 合计。"""
        per_food = []
        tot_kcal = tot_p = tot_f = tot_c = tot_w = 0.0
        for idx, w in items:
            r = self.compute(idx, w)
            per_food.append(r)
            tot_kcal += r.get("kcal", 0.0)
            tot_p += r.get("protein_g", 0.0)
            tot_f += r.get("fat_g", 0.0)
            tot_c += r.get("carbs_g", 0.0)
            tot_w += r.get("weight_g", 0.0)
        return {
            "items": per_food,
            "total": {
                "weight_g": round(tot_w, 1), "kcal": round(tot_kcal, 1),
                "protein_g": round(tot_p, 1), "fat_g": round(tot_f, 1), "carbs_g": round(tot_c, 1),
            },
        }

    def to_json(self, result):
        return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    nq = NutritionQuerier()
    # 自测：麻婆豆腐 220g
    print("=== 单食物 麻婆豆腐 220g ===")
    print(nq.to_json(nq.compute(0, 220)))
    # 一餐
    print("\n=== 一餐: 麻婆豆腐220g + 米饭150g ===")
    print(nq.to_json(nq.compute_meal([(0, 220), (16, 150)])))
