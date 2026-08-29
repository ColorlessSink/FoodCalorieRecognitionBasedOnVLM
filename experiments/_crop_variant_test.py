'''
experiments/_crop_variant_test.py — 混合盘组件裁剪图的"背景处理"变体测试（诊断脚本）
================================================
为什么测：混合盘组件识别 Top-1 78.2% 低于单食物 83.67%，怀疑裁剪图里
的"白盘背景"是域差来源（LoRA 训练口径是整图食物照片）。
变体：raw / white / gray / blur —— 检测 mask 之外的像素分别置白/置灰/强模糊。
每变体单独进程跑（显卡 6GB，同进程连续加载多副本会触发 OSError 1455）。
'''
import os
import sys
import pickle
import numpy as np
import cv2
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "raw"
    cfg = load_config()

    # 先加载模型（经验顺序：先模型后缓存，避免页文件峰值叠加）
    from models.food_recognizer import FoodRecognizer
    adapter = os.path.join(ROOT, cfg["recognition"]["lora"]["adapter_dir_50"])
    mode = "lora" if (os.path.isdir(adapter) and os.listdir(adapter)) else "zero_shot"
    rec = FoodRecognizer(cfg=cfg, mode=mode)

    with open(os.path.join(ROOT, "results", "_mixed_crops2.pkl"), "rb") as f:
        cache2 = pickle.load(f)

    def _fit(mask, crop):
        if mask.shape[:2] != crop.shape[:2]:
            return cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask

    if variant == "raw":
        imgs = [Image.fromarray(cv2.cvtColor(c[0], cv2.COLOR_BGR2RGB)) for c in cache2]
    elif variant == "white":
        def f(c):
            out = c[0].copy(); out[_fit(c[1], c[0]) == 0] = (255, 255, 255)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        imgs = [f(c) for c in cache2]
    elif variant == "gray":
        def f(c):
            out = c[0].copy(); out[_fit(c[1], c[0]) == 0] = (128, 128, 128)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        imgs = [f(c) for c in cache2]
    elif variant == "blur":
        def f(c):
            bl = cv2.GaussianBlur(c[0], (0, 0), 25)
            m = _fit(c[1], c[0])[..., None] > 0
            out = np.where(m, c[0], bl)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        imgs = [f(c) for c in cache2]
    else:
        raise SystemExit(f"未知变体 {variant}")

    gts = [c[2] for c in cache2]
    preds, confs, topk = rec.recognize(imgs)
    acc = float(np.mean([p == g for p, g in zip(preds, gts)]))
    top5 = float(np.mean([g in [t[0] for t in tk[:5]] for tk, g in zip(topk, gts)]))
    print(f"[{variant:6s}] mode={mode}  Top-1 {acc*100:.1f}%  Top-5 {top5*100:.1f}%")


if __name__ == "__main__":
    main()
