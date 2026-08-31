'''
50 类 LoRA 微调训练
---
背景：小作业只在 20 类上训了 LoRA adapter（92.5%/99.25%）。50 类类别更多、类间更细
（如鱼香肉丝 vs 回锅肉 vs 爆炒腰花 都是炒肉系），零样本 78.33% 已达标但有余量。
本脚本在 50 类训练集上用 InfoNCE 对比学习微调 LoRA，验证"少样本域适应"创新点
能否在 50 类上进一步提升，并产出 results/lora_adapter_50 供 ablation/识别模块复用

设计（沿用小作业 lora_train.py 验证过的方法，仅把数据/类别口径切到 50 类）：
  - 冻结 Chinese-CLIP 全部参数，只在文本塔/视觉塔的 Q、V 注入 LoRA(r=8, α=16)
  - InfoNCE 双向 loss（图→文 + 文→图），logit_scale 用模型自带的温度
  - 每 epoch 评测 val，保留 val Top-1 最高的 adapter
  - CLS+projection 兼容：取 image_embeds/text_embeds（ChineseCLIPModel 直接给对齐嵌入）
'''
import os, sys, time, json
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import peft as pf

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import ROOT, load_config, load_clip, load_classes, device_of

TEMPLATE = "一张中餐菜品{c}的照片"   # 与识别模块主模板一致（50 类实测最优）
EPOCH = 6                            # 50 类 1500 图，6 epoch 足够收敛且省时
BATCH_SIZE = 16


def collate_fn(batch, processor):
    pics = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    procs = processor(text=texts, images=pics, padding=True, return_tensors="pt")
    return procs, labels


class FoodDataset50(Dataset):
    # 从 dataset_50cls/train.csv 读 (path,label)，套模板成文本
    def __init__(self, data_dir, split, template, transform=None):
        self.data_dir = data_dir
        names_zh, class_idx, _ = load_classes(data_dir)
        self.names_zh = names_zh
        self.texts = [template.format(c=n) for n in names_zh]

        df = pd.read_csv(os.path.join(ROOT, data_dir, f"{split}.csv"))
        self.samples = []
        for _, row in df.iterrows():
            p = os.path.join(ROOT, data_dir, str(row["path"]).replace("\\", "/"))
            label = int(row["label"])
            self.samples.append((p, self.texts[label], label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pic = Image.open(self.samples[i][0]).convert("RGB")
        if self.transform is not None:
            pic = self.transform(pic)
        return pic, self.samples[i][1], self.samples[i][2]


@torch.no_grad()
def evaluate(model, processor, data_dir, split, names_zh, template, device, batch_size=16):
    # 零开销评估：文本特征算一次，图特征批量算，余弦相似度取 argmax
    model.eval()
    texts = [template.format(c=n) for n in names_zh]
    t_in = processor(text=texts, padding=True, return_tensors="pt").to(device)
    t_out = model.get_text_features(**t_in)
    t_feats = t_out.text_embeds if (hasattr(t_out, "text_embeds") and t_out.text_embeds is not None) \
        else model.text_projection(t_out.last_hidden_state[:, 0, :])
    t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)

    df = pd.read_csv(os.path.join(ROOT, data_dir, f"{split}.csv"))
    paths = [os.path.join(ROOT, data_dir, str(p).replace("\\", "/")) for p in df["path"]]
    labels = df["label"].tolist()

    correct = 0
    top5_hit = 0
    n = 0
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        v_in = processor(images=imgs, return_tensors="pt").to(device)
        v_out = model.get_image_features(**v_in)
        v_feats = v_out.image_embeds if (hasattr(v_out, "image_embeds") and v_out.image_embeds is not None) \
            else model.visual_projection(v_out.last_hidden_state[:, 0, :])
        v_feats = v_feats / v_feats.norm(dim=-1, keepdim=True)
        logits = 100.0 * (v_feats @ t_feats.T)     # [B, C]
        preds = logits.argmax(dim=-1).tolist()
        top5 = logits.topk(min(5, logits.shape[1]), dim=-1).indices.tolist()
        for j, lab in enumerate(labels[i:i+batch_size]):
            if preds[j] == lab:
                correct += 1
            if lab in top5[j]:
                top5_hit += 1
            n += 1
    return correct / n, top5_hit / n


def train():
    cfg = load_config()
    data_dir = cfg["project"]["data_dir"]      # dataset_50cls
    device = device_of(cfg)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"配置 data_dir={data_dir} device={device} epoch={EPOCH} bs={BATCH_SIZE}")

    model, processor = load_clip(cfg)
    model = model.to(device)

    names_zh, class_idx, _ = load_classes(data_dir)
    print(f"共 {len(names_zh)} 类")

    dataset = FoodDataset50(data_dir, "train", TEMPLATE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                            collate_fn=lambda b: collate_fn(b, processor), num_workers=0)

    # 注入 LoRA：冻结主体，只在 Q/V 注入低秩
    model.requires_grad_(False)
    config = pf.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                           target_modules=["query", "value", "q_proj", "v_proj"])
    model = pf.get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=0.01)

    adapter_dir = os.path.join(ROOT, "results", "lora_adapter_50")
    os.makedirs(adapter_dir, exist_ok=True)
    best_acc = 0.0
    history = []
    t_start = time.time()

    for ep in range(EPOCH):
        model.train()
        epoch_loss = 0.0
        nb = 0
        for (inputs, _labels) in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs)
            img_feats = output.image_embeds
            txt_feats = output.text_embeds
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (img_feats @ txt_feats.T)
            labels = torch.arange(logits.shape[0]).to(device)
            loss = (torch.nn.functional.cross_entropy(logits, labels) +
                    torch.nn.functional.cross_entropy(logits.T, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        avg_loss = epoch_loss / nb

        # val 评测
        top1, top5 = evaluate(model, processor, data_dir, "val", names_zh, TEMPLATE, device, BATCH_SIZE)
        history.append({"epoch": ep+1, "train_loss": round(avg_loss, 4),
                        "val_top1": round(top1*100, 2), "val_top5": round(top5*100, 2)})
        print(f"Epoch {ep+1}/{EPOCH}: loss={avg_loss:.4f}  val_top1={top1*100:.2f}%  val_top5={top5*100:.2f}%")

        if top1 > best_acc:
            best_acc = top1
            model.save_pretrained(adapter_dir)
            print(f"  ↑ 新最佳，adapter 已保存到 {adapter_dir}")

    dt = time.time() - t_start
    # 训完用最佳 adapter 在 test 上评一遍
    print(f"\n训练完成 用时 {dt/60:.1f} min，最佳 val_top1={best_acc*100:.2f}%")
    # 重新加载最佳 adapter 评 test
    del model
    torch.cuda.empty_cache()
    base, proc2 = load_clip(cfg)
    base = base.to(device)
    peft_model = pf.PeftModel.from_pretrained(base, adapter_dir).to(device)
    peft_model.eval()
    names_zh2, _, _ = load_classes(data_dir)
    test_top1, test_top5 = evaluate(peft_model, proc2, data_dir, "test", names_zh2, TEMPLATE, device, BATCH_SIZE)
    summary = {
        "method": "LoRA r=8 alpha=16 (50类)",
        "data_dir": data_dir,
        "n_classes": len(names_zh),
        "template": TEMPLATE,
        "epochs": EPOCH,
        "best_val_top1": round(best_acc*100, 2),
        "test_top1": round(test_top1*100, 2),
        "test_top5": round(test_top5*100, 2),
        "train_minutes": round(dt/60, 1),
        "history": history,
    }
    out = os.path.join(ROOT, "results", "lora_50_test_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n========== 50 类 LoRA 测试结果 ==========")
    print(f"Top-1: {test_top1*100:.2f}%   Top-5: {test_top5*100:.2f}%")
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    train()
