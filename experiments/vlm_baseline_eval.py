'''
VLM 基线评估：Qwen2-VL-2B-Instruct 零样本分类（600 张 test 图）
---
作业要求"至少对比 2 种 VLM 基线（如 CLIP vs. LLaVA，或 CLIP vs. GPT-4V API）"。
本脚本在服务器（环境 B，RTX 3090）上运行 Qwen2-VL-2B 生成式分类，与
Chinese-CLIP 的判别式基线（results/recognition_baseline.json）同口径对比。

设计要点：
  1. 与 baseline_eval.py 完全同数据集（dataset_50cls/test 600 张）、同标签（classes_50.csv）；
  2. 生成式判分：把 50 类菜名列进 prompt，模型只允许从中选一个（约束解码不适用，
     用提示工程 + 输出后处理做菜名归一，归一失败的样本记为错且保留原文供分析）；
  3. Top-5：单次生成拿不到分布，改为让模型输出"最可能的 5 个"（一次前向解决，避免 600×50
     次打分的开销；生成式 Top-5 语义与判别式 Top-5 略有差异，报告里如实说明）；
  4. 服务器离线运行：模型已通过 hf-mirror 下载到本地缓存，transformers 4.46.3 原生支持
     Qwen2-VL（无需 trust_remote_code）。

产出：vlm_baseline_result.json（拷回本地后并入 results/recognition_baseline.json）
'''
import os, sys, csv, json, time, re

sys.stdout.reconfigure(encoding="utf-8")

# ---------- 数据 ----------
BASE = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE, "test.csv")
CLASSES_CSV = os.path.join(BASE, "classes_50.csv")
OUT_JSON = os.path.join(BASE, "vlm_baseline_result.json")

# 服务器 GPU 编号按空闲显存选，默认 0（7.6G 已用，Qwen2-VL-2B bf16 约 5-6G 够用）
GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES_IDX", "0"))


def load_classes():
    # classes_50.csv 实际格式：idx, zh, en, orig_id（带表头）
    rows = list(csv.reader(open(CLASSES_CSV, encoding="utf-8-sig")))
    classes = [(int(r[0]), r[1].strip()) for r in rows[1:] if len(r) >= 2]
    classes.sort(key=lambda t: t[0])
    assert len(classes) == 50, f"类别数 {len(classes)} != 50"
    return [c[1] for c in classes]


def load_test():
    rows = list(csv.reader(open(TEST_CSV, encoding="utf-8-sig")))
    data = rows[1:]
    items = []
    for r in data:
        rel = r[1].replace("\\", "/")
        items.append((os.path.join(BASE, "test_imgs", rel), int(r[2])))
    return items


def normalize_pred(text, class_names):
    # 生成输出 → 类别 idx；失败返回 None（记错但不中断）
    t = text.strip()
    # 直接全等
    if t in class_names:
        return class_names.index(t)
    # 去引号/句号/空白
    t_clean = t.strip("\"'。！! \n\t ")
    if t_clean in class_names:
        return class_names.index(t_clean)
    # 子串匹配（模型可能输出"这是一张麻婆豆腐的照片"）
    for i, c in enumerate(class_names):
        if c in t:
            return i
    return None


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from PIL import Image

    class_names = load_classes()
    items = load_test()
    print(f"类别数 {len(class_names)}，测试图 {len(items)} 张，GPU {GPU_ID}")

    model_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
    )
    assert os.path.isdir(model_dir), f"模型目录不存在: {model_dir}"

    processor = AutoProcessor.from_pretrained(model_dir, max_pixels=768*28*28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    # 50 类列表进 prompt，要求模型只回菜名（最可能 1 个 + 备选 4 个）
    cls_listing = "、".join(class_names)
    PROMPT = (
        "你是中餐食物识别专家。下图是一道中餐菜品，请从下面 50 个候选菜名中选出它。\n"
        f"候选菜名：{cls_listing}\n"
        "输出格式（严格两行）：\n"
        "答案：<最可能的菜名>\n"
        "备选：<按可能性排序的第二到第五个菜名，用、分隔>\n"
        "要求：菜名必须逐字来自候选列表，不要输出其他任何内容。"
    )

    top1_hit, top5_hit = 0, 0
    records = []
    t0 = time.time()
    for n, (path, label) in enumerate(items):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  [warn] 打不开 {path}: {e}")
            records.append({"path": path, "label": label, "raw": "", "pred": None,
                            "top5_raw": [], "top1": False, "top5": False})
            continue
        messages = [
            {"role": "user",
             "content": [
                 {"type": "image", "image": image},
                 {"type": "text", "text": PROMPT},
             ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        # 去掉 prompt 部分
        gen = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)[0]
        # 解析两行
        ans_line, alt_line = "", ""
        for line in gen.splitlines():
            line = line.strip()
            if line.startswith("答案"):
                ans_line = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("备选"):
                alt_line = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        pred = normalize_pred(ans_line, class_names)
        # 备选拼 top5（答案 + 备选前 4）
        top5_names = []
        if pred is not None:
            top5_names.append(class_names[pred])
        for c in [x.strip() for x in alt_line.split("、") if x.strip()]:
            if c in class_names and c not in top5_names:
                top5_names.append(c)
            if len(top5_names) >= 5:
                break
        t1 = (pred == label)
        t5 = (label in [class_names.index(c) for c in top5_names]) if top5_names else False
        top1_hit += int(t1)
        top5_hit += int(t5)
        records.append({"path": path, "label": label, "raw": gen[:200],
                        "pred": pred, "pred_name": class_names[pred] if pred is not None else None,
                        "top5_names": top5_names, "top1": bool(t1), "top5": bool(t5)})
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(items)}  Top-1={top1_hit/(n+1)*100:.2f}%  Top-5={top5_hit/(n+1)*100:.2f}%  ({el:.0f}s)")

    dt = time.time() - t0
    result = {
        "method": "qwen2_vl_2b_zeroshot",
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "top1": round(top1_hit / len(items) * 100, 2),
        "top5": round(top5_hit / len(items) * 100, 2),
        "n": len(items),
        "seconds": round(dt, 1),
        "hardware": "RTX 3090 24GB, bf16",
        "note": "生成式分类（50 类候选约束 prompt + 输出归一），与 Chinese-CLIP 判别式同测试集同标签口径；Top-5 为模型自报备选前 4 + 答案，语义与判别式 top-5 略有差异",
        "records": records,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nQwen2-VL-2B 零样本  Top-1={result['top1']}%  Top-5={result['top5']}%  ({dt:.0f}s)")
    print(f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
