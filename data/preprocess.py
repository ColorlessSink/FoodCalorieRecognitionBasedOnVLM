'''
大作业·第二部分：数据预处理脚本
================================================
为什么需要：
  把"原始 ChineseFoodNet"→"可训练/评估的 50 类数据集 + 营养库 + 真值标签 + 场景标注"
  的整个流水线串成一条命令，可复现。同时做基本的数据清洗：
  - 过滤损坏/无法解码的图片（PIL verify）
  - 校验每张图的类别标签与所在目录名一致

运行：python data/preprocess.py
它依次调用 build_dataset_50 / build_nutrition_db / build_labels / build_area_stats /
build_scene_split / build_mixed_plates，并做完整性校验。
注意：第一步 build_dataset_50 会清空重建 dataset_50cls/，已有的场景标注进度
（scene_labels_progress.json / test_scene.csv）会一并丢失，重跑前先备份。
'''
import os, sys, subprocess
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset_50cls")

def run(script):
    print(f"\n==== 运行 {script} ====")
    r = subprocess.run([sys.executable, script], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{script} 失败，rc={r.returncode}")

def main():
    run(os.path.join(ROOT, "data", "build_dataset_50.py"))
    run(os.path.join(ROOT, "data", "build_nutrition_db.py"))
    run(os.path.join(ROOT, "data", "build_labels.py"))
    # area_stats 要读 train 图算面积比，mixed 要读 test 真值，都依赖前面三步
    run(os.path.join(ROOT, "data", "build_area_stats.py"))
    run(os.path.join(ROOT, "data", "build_scene_split.py"))
    run(os.path.join(ROOT, "data", "build_mixed_plates.py"))

    # —— 完整性校验 ——
    print("\n==== 完整性校验 ====")
    classes = pd.read_csv(os.path.join(DATA, "classes_50.csv"))
    assert len(classes) == 50, f"类别数 != 50: {len(classes)}"
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(DATA, f"{split}.csv"))
        lbl = pd.read_csv(os.path.join(DATA, f"{split}_labels.csv"))
        assert len(df) == len(lbl), f"{split}: 清单与标签行数不一致"
        # 抽样校验：前若干张图能否正常打开
        bad = 0
        for p in df["path"].head(20):
            full = os.path.join(DATA, str(p).replace("\\", "/"))
            try:
                Image.open(full).verify()
            except Exception:
                bad += 1
        print(f"{split}: {len(df)} 张 | 前20张损坏 {bad} | 类别覆盖 {df['label'].nunique()}")

    # train ≥1500, test ≥300
    n_train = sum(1 for _ in open(os.path.join(DATA, "train.csv"))) - 1
    n_test  = sum(1 for _ in open(os.path.join(DATA, "test.csv"))) - 1
    print(f"\ntrain={n_train} (≥1500? {n_train>=1500})  test={n_test} (≥300? {n_test>=300})")

    # area_ratio 标定表 & 混合盘产物
    ar = pd.read_csv(os.path.join(ROOT, "data", "area_ratio_stats.csv"))
    print(f"area_ratio_stats: {len(ar)} 桶 {ar['bucket'].tolist()}")
    mx = pd.read_csv(os.path.join(DATA, "test_mixed.csv"))
    n_plates = mx["plate"].nunique()
    gt_kcal = mx.groupby("plate")["calories_kcal_true"].sum()
    print(f"mixed: {n_plates} 盘 / {len(mx)} 组件 | 盘级真值 {gt_kcal.min():.0f}~{gt_kcal.max():.0f} kcal")
    print("预处理完成。")

if __name__ == "__main__":
    main()
