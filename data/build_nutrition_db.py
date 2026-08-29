'''
大作业·第二部分：构建营养成分数据库 nutrition_db.csv
================================================
为什么需要：
  大作业要求"根据识别出的食物类别查询每100g的卡路里/蛋白质/脂肪/碳水"，
  并结合分量算总卡路里。ChineseFoodNet 没有营养标注，所以需要自建营养库。

数据来源与口径：
  各菜品"每100g"的营养值为参考《中国食物成分表》及常见膳食数据库的代表性估算值，
  非精确测量。本作业用其做"识别→查表→按分量折算"的管线演示与一致性评估，
  不作为临床营养建议。
  density(g/cm³) 列用于分量估计模块的"面积×厚度×密度"重量估算先验。

输出：data/nutrition_db.csv，列 = idx,zh,en,kcal_per100g,protein_g,fat_g,carbs_g,density
'''
import os, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "nutrition_db.csv")

# (idx, zh, en, kcal/100g, 蛋白g, 脂肪g, 碳水g, 密度g/cm³)
# 密度先验：固体主食/肉类 ~1.0-1.1，叶菜 ~0.6-0.8，汤水 ~1.0，油炸 ~0.7-0.9（孔隙）
NUTR = [
    (0,  "麻婆豆腐", "Mapo Tofu", 150, 12, 9,  6,  0.95),
    (1,  "宫保鸡丁", "Kung Pao Chicken", 200, 15, 13, 8, 1.00),
    (2,  "回锅肉",   "Double Cooked Pork", 250, 14, 20, 5, 1.00),
    (3,  "鱼香肉丝", "Yu-Shiang Shredded Pork", 180, 12, 12, 9, 1.00),
    (4,  "水煮鱼",   "Fish in Hot Chili Oil", 180, 16, 11, 5, 1.00),
    (5,  "鱼香茄子", "Yu-Shiang Eggplant", 160, 3,  12, 12, 0.80),
    (6,  "酸辣土豆丝","Hot and Sour Potato", 110, 2,  6,  13, 0.90),
    (7,  "西红柿炒蛋","Tomato and Egg", 120, 6,  8,  6,  0.95),
    (8,  "地三鲜",   "Di San Xian", 130, 3,  8,  12, 0.85),
    (9,  "蚝油生菜", "Oyster Sauce Lettuce", 60,  2,  3,  6,  0.60),
    (10, "红烧肉",   "Braised Pork", 300, 14, 26, 6, 1.05),
    (11, "糖醋排骨", "Sweet and Sour Spareribs", 280, 16, 20, 10, 1.05),
    (12, "梅菜扣肉", "Pork with Salted Vegetable", 320, 12, 28, 8, 1.00),
    (13, "京酱肉丝", "Sweet Bean Pork", 220, 16, 14, 8, 1.00),
    (14, "饺子",     "Dumplings", 250, 9,  12, 28, 1.05),
    (15, "包子",     "Steamed Stuffed Bun", 230, 9,  8,  30, 0.95),
    (16, "扬州炒饭", "Yangzhou Fried Rice", 180, 6,  7,  24, 1.00),
    (17, "炸酱面",   "Zhajiang Noodles", 180, 7,  6,  25, 1.00),
    (18, "葱爆羊肉", "Scallion Lamb", 220, 16, 15, 6, 1.00),
    (19, "香辣小龙虾","Spicy Crayfish", 110, 18, 3,  2,  1.00),
    (20, "家常豆腐", "Home style Tofu", 130, 9,  8,  6,  0.95),
    (21, "薯条",     "French Fries", 300, 4,  15, 40, 0.70),
    (22, "蚝油西兰花","Broccoli with Oyster Sauce", 70,  3,  3,  7,  0.65),
    (23, "炸藕盒",   "Deep Fried Lotus Root", 230, 6,  15, 22, 0.85),
    (24, "凉拌西红柿","Tomato Salad", 40,  1,  0,  9,  0.95),
    (25, "花生",     "Peanut", 570, 25, 49, 16, 0.85),
    (26, "炒花菜",   "Fried Cauliflower", 80,  3,  4,  8,  0.70),
    (27, "蒸蛋羹",   "Steamed Egg Custard", 90,  8,  6,  3,  1.00),
    (28, "可乐鸡翅", "Cola Chicken Wings", 200, 16, 12, 8, 1.05),
    (29, "辣子鸡",   "Spicy Chicken", 250, 18, 18, 6, 1.00),
    (30, "爆炒腰花", "Scalloped Kidneys", 160, 16, 9,  4,  1.05),
    (31, "红烧牛肉", "Braised Beef", 200, 22, 10, 5, 1.05),
    (32, "酸菜鱼",   "Fish with Pickled Cabbage", 120, 14, 5,  5,  1.00),
    (33, "糖醋鱼",   "Sweet and Sour Fish", 200, 16, 10, 12, 1.00),
    (34, "剁椒鱼头", "Fish Head with Chili", 130, 16, 6,  4,  1.00),
    (35, "麻辣虾",   "Spicy Shrimp", 130, 20, 4,  3,  1.00),
    (36, "螃蟹",     "Crab", 110, 18, 3,  2,  1.00),
    (37, "小笼包",   "Steamed Stuffed Bun (Small)", 260, 10, 12, 30, 0.95),
    (38, "肉夹馍",   "Chinese Hamburger", 280, 10, 14, 30, 0.95),
    (39, "酸辣粉",   "Hot and Sour Rice Noodles", 150, 4,  6,  22, 1.00),
    (40, "凉面",     "Cold Noodles", 170, 5,  5,  28, 1.00),
    (41, "炒面",     "Fried Noodles", 180, 6,  7,  25, 1.00),
    (42, "煎饺",     "Fried Dumplings", 260, 9,  13, 28, 0.95),
    (43, "白粥",     "Rice Porridge", 50,  1,  0,  11, 1.00),
    (44, "米饭",     "Rice", 130, 3,  0,  28, 1.00),
    (45, "炒菠菜",   "Sauteed Spinach", 70,  3,  4,  5,  0.60),
    (46, "蛋挞",     "Egg Tart", 320, 6,  20, 35, 0.80),
    (47, "面包",     "Bread", 280, 8,  6,  50, 0.50),
    (48, "双皮奶",   "Double Skin Milk", 130, 5,  6,  14, 1.00),
    (49, "冰淇淋",   "Ice Cream", 180, 4,  8,  22, 0.90),
]

COLS = ["idx", "zh", "en", "kcal_per100g", "protein_g", "fat_g", "carbs_g", "density"]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(NUTR, columns=COLS)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已写入 {OUT}  | {len(df)} 类")
    print(df[["zh", "kcal_per100g", "protein_g", "fat_g", "carbs_g", "density"]].to_string())

if __name__ == "__main__":
    main()
