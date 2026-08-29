'''
大作业·第二部分：测试集 3 种场景标注（图像统计代理）
输出 dataset_50cls/test_scene.csv：path,label,scene in {standard,real,challenge}
'''
import os, numpy as np, pandas as pd
from PIL import Image
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dataset_50cls")

def image_stats(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    arr = np.array(img.resize((128, 128)))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    v_mean = float(hsv[:, :, 2].mean()) / 255.0
    s_mean = float(hsv[:, :, 1].mean()) / 255.0
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ar = max(w, h) / (min(w, h) + 1e-6)
    return v_mean, s_mean, blur, ar

def main():
    df = pd.read_csv(os.path.join(DATA, "test.csv"))
    paths = [os.path.join(DATA, str(p).replace("\\", "/")) for p in df["path"]]

    stats = [image_stats(p) for p in paths]
    v  = np.array([s[0] for s in stats])
    sv = np.array([s[1] for s in stats])
    bl = np.array([s[2] for s in stats])
    ar = np.array([s[3] for s in stats])

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)
    score = (1 - norm(v)) * 0.35 + (1 - norm(sv)) * 0.25 + (1 - norm(np.log1p(bl))) * 0.25 + norm(ar) * 0.15

    df["_score"] = score
    scene = [""] * len(df)
    for cls, grp in df.groupby("label"):
        ordered = grp.sort_values("_score").index.tolist()
        n = len(ordered); n_lo = n // 3; n_hi = 2 * n // 3
        for pos, gi in enumerate(ordered):
            rpos = df.index.get_loc(gi)
            scene[rpos] = "standard" if pos < n_lo else ("real" if pos < n_hi else "challenge")

    df["scene"] = scene
    df.drop(columns=["_score"]).to_csv(os.path.join(DATA, "test_scene.csv"), index=False, encoding="utf-8-sig")
    from collections import Counter
    print("场景分布:", dict(Counter(scene)))
    print("已写入", os.path.join(DATA, "test_scene.csv"))

if __name__ == "__main__":
    main()
