'''
生成每张图像的卡路里真值标签脚本
---
ChineseFoodNet 无分量/卡路里标注，采用"合成真值"策略：
  真值重量(g) = 按类别高斯先验 N(mu_cls, sigma_cls) 采样，以"图像文件名的稳定哈希"为种子，
                保证 同一张图每次得到的真值一致（可复现），且与估计器输入（图像像素）完全无关
  真值卡路里   = 重量 × nutrition_db[idx].kcal_per100g / 100
这样真值只依赖 (文件名, 类别先验, 营养库)，与"估计器如何看图"解耦，评估公平

输出：dataset_50cls/<split>_labels.csv  列 = path,label(类别idx),name,weight_g(true),calories_kcal(true)
'''
import os, hashlib, math
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset_50cls")
NUTR = os.path.join(ROOT, "data", "nutrition_db.csv")

# 每类重量先验均值/标准差（克）——"单份成品"的自然重量波动
# 口径：一张图里拍到的是"一份"（一碗/一盘/一个），不是"任意可能份量"
#   - 主食(米饭/炒饭/面条)：餐厅单份 240~360g → μ=300, σ=36
#   - 炒菜/肉类：单盘 180~280g → μ=220, σ=32
#   - 小吃/甜点(饺子/包子/冰淇淋)：单份 110~190g → μ=150, σ=26
#   - 凉拌/叶菜/小份：单份 80~160g → μ=120, σ=22
PORTION_PRIOR = {
    "staple": (300, 36),  # 米饭/炒饭/面条/粥
    "dish":   (220, 32),  # 炒菜/炖菜/肉
    "snack":  (150, 26),  # 饺子/包子/小吃/甜点/冰淇淋
    "light":  (120, 22),  # 凉拌/叶菜/小份
}

# 按类别归类到先验桶（idx -> 桶）
def bucket_of(idx, zh):
    staples = {16,17,37,38,39,40,41,14,15,42,43,44,48}      # 炒饭/面/饺子/包子/粥/米饭/酸辣粉/肉夹馍/双皮奶
    snacks  = {21,23,25,46,47,49,20,27}                     # 薯条/炸藕盒/花生/蛋挞/面包/冰淇淋/家常豆腐/蒸蛋
    lights  = {9,22,24,26,45,19,36}                         # 蚝油生菜/西兰花/凉拌西红柿/花菜/菠菜/小龙虾/螃蟹
    if idx in staples: return "staple"
    if idx in snacks:  return "snack"
    if idx in lights:  return "light"
    return "dish"

def hash_seed(path):
    # 对相对路径取稳定哈希 → [0,1) 浮点，作采样种子
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF     # 0~1

def gaussian_from_hash(u, mu, sigma):
    # Box-Muller：把均匀分布转成标准正态。只用单个 u，近似 N(mu,sigma) 且夹在 ±3σ
    u1 = max(u, 1e-9)     # u1 避免为0
    u2 = (u * 2654435761 % 1.0)
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    val = mu + sigma * z
    return max(mu - 3 * sigma, min(mu + 3 * sigma, val))     # 裁到 ±3σ

def main():
    nutr = pd.read_csv(NUTR)
    kcal_map = {r["idx"]: r["kcal_per100g"] for _, r in nutr.iterrows()}
    name_map = {r["idx"]: r["zh"] for _, r in nutr.iterrows()}

    for split in ("train", "val", "test"):
        split_csv = os.path.join(DATA, f"{split}.csv")
        if not os.path.exists(split_csv):
            print(f"跳过 {split}（找不到 {split_csv}）")
            continue
        df = pd.read_csv(split_csv)
        rows = []
        for _, r in df.iterrows():
            idx = int(r["label"])
            rel = str(r["path"]).replace("\\", "/")
            mu, sigma = PORTION_PRIOR[bucket_of(idx, name_map[idx])]
            u = hash_seed(rel)
            weight = gaussian_from_hash(u, mu, sigma)
            weight = round(weight, 1)
            kcal = round(weight * kcal_map[idx] / 100, 1)
            rows.append({
                "path": rel, "label": idx, "name": name_map[idx],
                "weight_g_true": weight, "calories_kcal_true": kcal,
            })
        out = pd.DataFrame(rows)
        out_path = os.path.join(DATA, f"{split}_labels.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"{split}: {len(out)} 张 -> {out_path}")
        print(f"   重量均值 {out['weight_g_true'].mean():.1f}g | 卡路里均值 {out['calories_kcal_true'].mean():.1f}kcal")

if __name__ == "__main__":
    main()
