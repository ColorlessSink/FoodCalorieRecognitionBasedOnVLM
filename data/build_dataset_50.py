'''
大作业·第二部分：构建 50 类中餐食物数据集
================================================
为什么需要这个脚本：
  小作业的 dataset_20cls 只有 20 类，而大作业要求"食物识别覆盖 ≥ 50 类（中餐 ≥ 20）"。
  ChineseFoodNet 共 208 类，我们从中选 50 类常见中餐，按 30(train)+6(val)+12(test)/类
  划分，得到 train=1500 / val=300 / test=600，满足"训练≥1500、测试≥300"。

设计要点：
  - 输出目录 dataset_50cls/，与 dataset_20cls/ 并存，互不影响（小作业脚本仍可用 20 类）。
  - 优先用 ChineseFoodNet 的 test/ 图片作为我们的测试集（这些图模型在预训练时没当训练集用过，
    更贴近"真实未见过的图"），不足部分从 train/ 补。
  - 文件夹结构沿用小作业约定：<split>/<idx>_<name>/<idx>_<原文件名>，保证 zero_shot.py 里
    "int(i.split('_')[0])" 解析标签的逻辑依然成立。
  - 同步生成 classes_50.csv（idx,zh,en,orig_id）和各 split 清单 csv。

运行：在项目根目录 `python data/build_dataset_50.py`
'''
import os, random, shutil
from collections import defaultdict
import pandas as pd

# 原始数据在"高扬"同学的目录下（与本项目同级）；脚本在 data/ 下，根目录再上一层
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.normpath(os.path.join(ROOT, "..", "高扬", "ChineseFood Net 3", "release_data"))
OUT = os.path.join(ROOT, "dataset_50cls")

PER_CLASS_TRAIN, PER_CLASS_VAL, PER_CLASS_TEST = 30, 6, 12   # 30:6:12 ≈ 7:1.4:2.6
SEED = 42

# 50 类：前 20 类与小作业完全一致（便于复用其 LoRA adapter 做 20 类对照），
# 后 30 类新增。元组 = (new_idx, 中文名, 英文名, ChineseFoodNet orig_id)
SEL = [
    # ---- 前 20 类（同小作业）----
    (0,  "麻婆豆腐", "Mapo Tofu", 0),
    (1,  "宫保鸡丁", "Kung Pao Chicken", 71),
    (2,  "回锅肉",   "Double Cooked Pork", 84),
    (3,  "鱼香肉丝", "Yu-Shiang Shredded Pork", 97),
    (4,  "水煮鱼",   "Fish in Hot Chili Oil", 112),
    (5,  "鱼香茄子", "Yu-Shiang Eggplant", 11),
    (6,  "酸辣土豆丝","Hot and Sour Potato", 5),
    (7,  "西红柿炒蛋","Tomato and Egg", 50),
    (8,  "地三鲜",   "Di San Xian", 9),
    (9,  "蚝油生菜", "Oyster Sauce Lettuce", 18),
    (10, "红烧肉",   "Braised Pork", 77),
    (11, "糖醋排骨", "Sweet and Sour Spareribs", 58),
    (12, "梅菜扣肉", "Pork with Salted Vegetable", 83),
    (13, "京酱肉丝", "Sweet Bean Pork", 92),
    (14, "饺子",     "Dumplings", 159),
    (15, "包子",     "Steamed Stuffed Bun", 145),
    (16, "扬州炒饭", "Yangzhou Fried Rice", 130),
    (17, "炸酱面",   "Zhajiang Noodles", 149),
    (18, "葱爆羊肉", "Scallion Lamb", 104),
    (19, "香辣小龙虾","Spicy Crayfish", 118),
    # ---- 新增 30 类 ----
    (20, "家常豆腐", "Home style Tofu", 1),
    (21, "薯条",     "French Fries", 10),
    (22, "蚝油西兰花","Broccoli with Oyster Sauce", 26),
    (23, "炸藕盒",   "Deep Fried Lotus Root", 27),
    (24, "凉拌西红柿","Tomato Salad", 29),
    (25, "花生",     "Peanut", 33),
    (26, "炒花菜",   "Fried Cauliflower", 43),
    (27, "蒸蛋羹",   "Steamed Egg Custard", 53),
    (28, "可乐鸡翅", "Cola Chicken Wings", 60),
    (29, "辣子鸡",   "Spicy Chicken", 70),
    (30, "爆炒腰花", "Scalloped Kidneys", 75),
    (31, "红烧牛肉", "Braised Beef", 78),
    (32, "酸菜鱼",   "Fish with Pickled Cabbage", 106),
    (33, "糖醋鱼",   "Sweet and Sour Fish", 108),
    (34, "剁椒鱼头", "Fish Head with Chili", 111),
    (35, "麻辣虾",   "Spicy Shrimp", 117),
    (36, "螃蟹",     "Crab", 127),
    (37, "小笼包",   "Steamed Stuffed Bun (Small)", 132),
    (38, "肉夹馍",   "Chinese Hamburger", 141),
    (39, "酸辣粉",   "Hot and Sour Rice Noodles", 150),
    (40, "凉面",     "Cold Noodles", 151),
    (41, "炒面",     "Fried Noodles", 158),
    (42, "煎饺",     "Fried Dumplings", 165),
    (43, "白粥",     "Rice Porridge", 171),
    (44, "米饭",     "Rice", 172),
    (45, "炒菠菜",   "Sauteed Spinach", 22),
    (46, "蛋挞",     "Egg Tart", 196),
    (47, "面包",     "Bread", 197),
    (48, "双皮奶",   "Double Skin Milk", 203),
    (49, "冰淇淋",   "Ice Cream", 204),
]


def read_split(path, has_label=True):
    """读 ChineseFoodNet 的 list 文件，每行 '相对路径 类别id'。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img = parts[0]
            lab = int(parts[1]) if has_label else None
            pairs.append((img, lab))
    return pairs


def main():
    assert os.path.isdir(SRC), f"找不到 ChineseFoodNet 原始数据：{SRC}"

    print("读取原始 list 文件 ...")
    train_pairs = read_split(os.path.join(SRC, "train_list.txt"))
    val_pairs   = read_split(os.path.join(SRC, "val_list.txt"))
    test_truth  = read_split(os.path.join(SRC, "test_truth_list.txt"))

    # 按 orig_id 聚合：(文件系统相对路径, 所在目录)
    # train_list / val_list 的图片都在 train/ 目录下（CFN 约定），test 在 test/
    by_class = defaultdict(list)
    for img, lab in train_pairs:
        by_class[lab].append(("train/" + img, "train"))
    for img, lab in val_pairs:
        by_class[lab].append(("train/" + img, "train"))   # val_list 图也在 train/ 下
    for img, lab in test_truth:
        by_class[lab].append(("test/" + img, "test"))

    # 清空输出目录
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, s), exist_ok=True)

    random.seed(SEED)
    records, classes_rows = [], []

    for new_idx, zh, en, orig_id in SEL:
        all_imgs = by_class[orig_id]
        random.shuffle(all_imgs)
        # 优先把 test/ 的图留给测试集
        test_pool  = [x for x in all_imgs if x[1] == "test"]
        train_pool = [x for x in all_imgs if x[1] == "train"]

        n_test = PER_CLASS_TEST
        n_val  = PER_CLASS_VAL
        n_train = PER_CLASS_TRAIN

        chosen = []
        # 先填 test
        chosen += [(p, "test") for p, _ in test_pool[:n_test]]
        if len(chosen) < n_test:
            need = n_test - len(chosen)
            chosen += [(p, "test") for p, _ in train_pool[:need]]
            train_pool = train_pool[need:]
        # 再填 val / train
        val_sel   = train_pool[:n_val];   train_pool = train_pool[n_val:]
        train_sel = train_pool[:n_train]
        chosen += [(p, "val")   for p, _ in val_sel]
        chosen += [(p, "train") for p, _ in train_sel]

        folder_name = f"{new_idx:02d}_{zh}"
        for fs_rel, target_split in chosen:
            src_path = os.path.join(SRC, fs_rel)
            dst_dir  = os.path.join(OUT, target_split, folder_name)
            os.makedirs(dst_dir, exist_ok=True)
            # 全局唯一文件名：<new_idx>_<原文件名>
            fname = f"{new_idx:02d}_{os.path.basename(fs_rel)}"
            shutil.copyfile(src_path, os.path.join(dst_dir, fname))
            rel_out = os.path.join(target_split, folder_name, fname)
            records.append((target_split, rel_out, new_idx))

        classes_rows.append((new_idx, zh, en, orig_id))
        print(f"[{new_idx:02d}] {zh:<8} -> train {n_train} / val {n_val} / test {n_test}  (源id={orig_id})")

    # 写类别表与各 split 清单
    pd.DataFrame(classes_rows, columns=["idx", "zh", "en", "orig_id"]).to_csv(
        os.path.join(OUT, "classes_50.csv"), index=False, encoding="utf-8-sig")
    for split in ("train", "val", "test"):
        sub = [(s, p, i) for (s, p, i) in records if s == split]
        pd.DataFrame(sub, columns=["split", "path", "label"]).to_csv(
            os.path.join(OUT, f"{split}.csv"), index=False, encoding="utf-8-sig")

    n_train = sum(1 for r in records if r[0] == "train")
    n_val   = sum(1 for r in records if r[0] == "val")
    n_test  = sum(1 for r in records if r[0] == "test")
    print(f"\n完成。输出: {OUT}")
    print(f"总图: {len(records)}  | train {n_train} / val {n_val} / test {n_test}  | 类别 {len(SEL)}")


if __name__ == "__main__":
    main()
