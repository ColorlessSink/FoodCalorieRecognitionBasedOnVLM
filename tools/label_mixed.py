'''
真实混合菜图的组件预标注（检测 → 裁剪 → CLIP 预标 → 人工复核）
---
背景：dataset_external/mixed_raw/ 的 40 张是真实食堂/家常混合菜（网图），
与合成盘不同：没有 GT 圆心、没有 GT 组件数、没有 GT 类别——什么标注都没有。
要把它们接入评估，必须先回答"图里有几样菜、各在哪、各是什么"

为什么"机器预标 + 人工看图复核"两步走，而不是纯人工标：
  纯人工在 40 张图上画框+分类，一张 2~5 分钟，太慢；
  MixedDetector + FoodRecognizer(few_shot) 已经能给出大差不差的候选框和类别，
  人只需要"看图确认/改错"，工作量从"画框+分类"降为"核对"

流程：
  ① MixedDetector.detect(img) → 区域（mask/bbox/cx/cy/area）
  ② MixedDetector.crop(img, region) → 组件裁剪图
  ③ FoodRecognizer(few_shot, 支持集口径与 mixed_eval.py 完全一致：
     train split 前 k=10 张/类，seed=67) 逐裁剪图识别 → top-5 候选
  ④ 输出：
     dataset_external/mixed_crops/<plate>_<comp>.jpg    裁剪图（人工复核用）
     dataset_external/mixed_labels.csv                 plate,comp,bbox,cx,cy,area,label,conf,top1..top5
     dataset_external/_review_sheet_<i>.jpg            复核用拼接大图（编号标在角上）
  ⑤ 人工（看图）把改错写回 mixed_labels.csv（label 列直接改，或删掉错框行）

复核约定（与 tools/label_scene.py 同思路）：
  CSV 是唯一真值；复核大图只是看图辅助。改完 CSV 删掉 _review_sheet_*.jpg 即可
'''
import os, sys, json, random

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import ROOT, load_config, load_classes
from models.food_recognizer import FoodRecognizer, gather_from_split
from models.mixed_detector import MixedDetector
from tools.utils import imread_unicode

EXT = os.path.join(ROOT, "dataset_external")
CROP_DIR = os.path.join(EXT, "mixed_crops")
SHEET_COLS = 5          # 复核大图每行列数
SHEET_THUMB = 240       # 复核大图缩略图边长


def build_recognizer(cfg):
    # few_shot 识别器，支持集口径与 mixed_eval.py 完全一致（train 前 k 张/类）
    random.seed(cfg["recognition"]["few_shot"]["seed"])
    k = cfg["recognition"]["few_shot"]["k_shot"]
    tr_paths, tr_labels = gather_from_split(cfg["project"]["data_dir"], "train")
    sup_paths, sup_labels = [], []
    for idx in load_classes(cfg["project"]["data_dir"])[1]:
        cls_paths = [p for p, l in zip(tr_paths, tr_labels) if l == idx][:k]
        sup_paths += cls_paths
        sup_labels += [idx] * len(cls_paths)
    rec = FoodRecognizer(cfg=cfg, mode="few_shot")
    rec.build_prototypes(sup_paths, sup_labels)
    return rec


def annotate_mixed(det, rec):
    # 40 张混合图 → 区域检测 + 逐组件识别。返回 DataFrame 行列表
    rows = []
    mixed_dir = os.path.join(EXT, "mixed_raw")
    files = sorted(f for f in os.listdir(mixed_dir) if f.endswith(".jpg"))
    from PIL import Image
    for fi, fn in enumerate(files, 1):
        img = imread_unicode(os.path.join(mixed_dir, fn))
        if img is None:
            print(f"  读不到 {fn}，跳过")
            continue
        regions = det.detect(img)
        H, W = img.shape[:2]
        # 区域裁剪 + 识别（一次 batch 所有裁剪图）
        crops, pil_crops = [], []
        for r in regions:
            crop, _ = MixedDetector.crop(img, r)
            crops.append(crop)
            pil_crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        preds, confs, topk_list = (rec.recognize(pil_crops) if pil_crops
                                   else ([], [], []))
        # 裁剪图落盘（复核用）
        os.makedirs(CROP_DIR, exist_ok=True)
        for ci, crop in enumerate(crops):
            cv2.imencode(".jpg", crop)[1].tofile(
                os.path.join(CROP_DIR, f"{fn[:-4]}_c{ci}.jpg"))
        for ci, r in enumerate(regions):
            x, y, w, h = r["bbox"]
            row = {
                "plate": f"mixed_raw/{fn}", "comp": ci,
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "cx": round(r["cx"], 1), "cy": round(r["cy"], 1),
                "area_frac": round(r["area"] / float(H * W), 4),
                "label": preds[ci], "conf": round(confs[ci], 3),
            }
            for ti, (tidx, tname, tscore) in enumerate(topk_list[ci][:5]):
                row[f"top{ti + 1}"] = f"{tidx}|{tname}|{tscore:.3f}"
            rows.append(row)
        print(f"  [{fi}/{len(files)}] {fn}: {len(regions)} 区域 -> "
              + ", ".join(f"{t[1]}({t[2]:.2f})" for t in topk_list[:0])  # 占位防呆
              + ("" if not regions else
                 ", ".join(r[f"top1"] for r in rows[-len(regions):])))
    return rows


def make_review_sheets(labels_df, names_zh):
    # 把所有裁剪图拼成复核大图：每格 = 裁剪图 + 角标（编号/预标类名/置信度）
    # 人工看图核对 '图里到底是什么菜' 是否等于 '预标类名'
    files = sorted(f for f in os.listdir(CROP_DIR) if f.endswith(".jpg"))
    per_sheet = SHEET_COLS * 6      # 每张大图 30 格
    sheet_paths = []
    meta = labels_df.set_index(["plate", "comp"])
    for si in range(0, len(files), per_sheet):
        batch = files[si:si + per_sheet]
        rows_n = (len(batch) + SHEET_COLS - 1) // SHEET_COLS
        canvas = np.full((rows_n * SHEET_THUMB, SHEET_COLS * SHEET_THUMB, 3),
                         24, np.uint8)
        for gi, fn in enumerate(batch):
            img = imread_unicode(os.path.join(CROP_DIR, fn))
            if img is None:
                continue
            th = cv2.resize(img, (SHEET_THUMB - 6, SHEET_THUMB - 26),
                            interpolation=cv2.INTER_AREA)
            r, c = divmod(gi, SHEET_COLS)
            y0, x0 = r * SHEET_THUMB + 3, c * SHEET_THUMB + 3
            canvas[y0:y0 + th.shape[0], x0:x0 + th.shape[1]] = th
            # 角标：编号 + 预标
            plate = f"mixed_raw/{fn.split('_c')[0]}.jpg"
            comp = int(fn.split("_c")[-1].split(".")[0])
            try:
                rec_row = meta.loc[(plate, comp)]
                tag = f"#{si+gi} {names_zh[int(rec_row['label'])]} {rec_row['conf']:.2f}"
            except KeyError:
                tag = f"#{si+gi} ?"
            cv2.putText(canvas, tag, (x0 + 4, y0 + SHEET_THUMB - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                        cv2.LINE_AA)
        out = os.path.join(EXT, f"_review_sheet_{si // per_sheet + 1}.jpg")
        cv2.imencode(".jpg", canvas)[1].tofile(out)
        sheet_paths.append(out)
    return sheet_paths


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    names_zh, _, _ = load_classes(cfg["project"]["data_dir"])
    det = MixedDetector(cfg)
    print("[1/3] 检测 + 预标（few_shot 支持集与 mixed_eval 同口径）...")
    rec = build_recognizer(cfg)
    rows = annotate_mixed(det, rec)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(EXT, "mixed_labels.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[2/3] 预标注写入 {out_csv}（{len(df)} 组件行，"
          f"{df['plate'].nunique()} 张图）")

    print("[3/3] 生成复核大图...")
    sheets = make_review_sheets(df, names_zh)
    for p in sheets:
        print(f"  {os.path.relpath(p, ROOT)}")
    print("\n复核方式：打开 _review_sheet_*.jpg，对照角标类名看图，"
          "发现错误改 mixed_labels.csv 的 label 列（改成 classes_50.csv 的 idx）。")


if __name__ == "__main__":
    main()
