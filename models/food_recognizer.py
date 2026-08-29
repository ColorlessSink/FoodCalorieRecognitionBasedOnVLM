'''
models/food_recognizer.py — 食物识别模块（模块1）
================================================
为什么这样设计：
  大作业模块1要求"基于 VLM 实现开放域食物识别，≥50 类，输出类别+置信度"。
  我们复用小作业验证过的 Chinese-CLIP（CLS+projection 兼容写法），把它包装成一个
  统一的 FoodRecognizer，内部支持三种后端：
    mode="zero_shot" : 文本模板候选 → 余弦相似度（基线）
    mode="few_shot"  : 每类 K 张支持集图特征求均作 prototype → 相似度
    mode="lora"      : 加载 LoRA adapter 微调后的权重 → 相似度
  对外只暴露 recognize(paths)->(labels, confs, topk)，下游分量/营养/智能体模块无需关心后端。

为什么把 ZeroShot 的逻辑搬过来而不是 import 小作业的：
  小作业的 ZeroShot 类硬编码了 dataset_20cls、device="cuda"、_LOCAL_SNAP 等，
  且它的 gather_pic 走目录扫描而非从 split.csv 读。大作业用 dataset_50cls 和
  split.csv 清单，且要在 20/50 类间切换、要被智能体当作库调用，所以重写一个
  无副作用的版本，保留 CLS+projection 的关键兼容逻辑（小作业踩过的坑）。

关键点（与 memory 一致）：
  get_text_features / get_image_features 在新版 transformers 里可能返回
  "非最终隐藏态"（只有 last_hidden_state）。必须手动取 [:,0,:] 再过
  text_projection / visual_projection，才能得到 512 维对齐嵌入。
'''
import os
import time
import torch
from PIL import Image
import pandas as pd

from models.common import load_config, load_clip, load_classes, device_of


class FoodRecognizer:
    def __init__(self, cfg=None, mode="zero_shot", data_dir=None, device=None):
        """
        mode  : "zero_shot" | "few_shot" | "lora"
        data_dir : "dataset_50cls" / "dataset_20cls"；None 时取 cfg.project.data_dir
        """
        self.cfg = cfg or load_config()
        self.mode = mode
        self.data_dir = data_dir or self.cfg["project"]["data_dir"]
        self.device = device or device_of(self.cfg)
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        # 加载模型
        self.model, self.processor = load_clip(self.cfg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 若是 lora 模式，注入 adapter
        if mode == "lora":
            self._load_lora_adapter()

        # 类别表
        self.names_zh, self.class_idx, self.classes_df = load_classes(self.data_dir)
        self.template = self.cfg["recognition"]["template"]

        # 文本特征 / prototype 懒加载（few_shot 需要先喂数据）
        self._text_feats = None
        self._prototypes = None

    # ---------------- LoRA ----------------
    def _load_lora_adapter(self):
        import peft as pf
        # 50 类用 50 的 adapter，20 类用 20 的
        is_50 = "50" in os.path.basename(self.data_dir)
        key = "adapter_dir_50" if is_50 else "adapter_dir_20"
        adapter_dir = self.cfg["recognition"]["lora"].get(key) or self.cfg["recognition"]["lora"]["adapter_dir_20"]
        adapter_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), adapter_dir) \
            if not os.path.isabs(adapter_dir) else adapter_dir
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter 不存在：{adapter_dir}。请先训练（见 lora_train.py）。"
            )
        self.model = pf.PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()

    # ---------------- 特征编码（CLS+projection 兼容）----------------
    @torch.no_grad()
    def encode_texts(self, texts):
        t_inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(self.device)
        out = self.model.get_text_features(**t_inputs)
        # 与 encode_images 同理：新版返回 Tensor（projection 后 512 维嵌入）
        if torch.is_tensor(out):
            feats = out
        elif hasattr(out, "text_embeds") and out.text_embeds is not None:
            feats = out.text_embeds
        else:
            cls = out.last_hidden_state[:, 0, :]
            feats = self.model.text_projection(cls)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_images(self, images, batch_size=None):
        """images: list[PIL.Image] 或 list[path]。返回 [N,512] 归一化特征。"""
        batch_size = batch_size or self.cfg["model"]["batch_size"]
        # 路径 → PIL
        pil = []
        for im in images:
            if isinstance(im, str):
                pil.append(Image.open(im).convert("RGB"))
            else:
                pil.append(im.convert("RGB") if im.mode != "RGB" else im)
        feats_all = []
        for i in range(0, len(pil), batch_size):
            batch = pil[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            # 三种 transformers 版本的返回形态都兼容：
            #   ① Tensor 且最后一维是 512（新版与旧版正常路径，projection 后嵌入）
            #   ② Tensor 且最后一维是序列长度（个别版本裸 last_hidden_state，
            #     形状 [N, seq, hidden] 三维）—— 取 CLS 再过 visual_projection
            #   ③ 对象带 image_embeds 属性（旧版返回 BaseModelOutput）
            if torch.is_tensor(out):
                feats = out if out.dim() == 2 else self.model.visual_projection(out[:, 0, :])
            elif hasattr(out, "image_embeds") and out.image_embeds is not None:
                feats = out.image_embeds
            else:
                cls = out.last_hidden_state[:, 0, :]
                feats = self.model.visual_projection(cls)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_all.append(feats)
        return torch.cat(feats_all, dim=0)

    # ---------------- 候选表示构建 ----------------
    def build_text_feats(self, template=None):
        """zero_shot / lora 模式：用模板生成每类文本特征。"""
        tpl = template or self.template
        texts = [tpl.format(c=n) for n in self.names_zh]
        self._text_feats = self.encode_texts(texts)
        return self._text_feats

    def build_prototypes(self, support_paths, support_labels, batch_size=None):
        """few_shot 模式：支持集图特征按类求均作 prototype。"""
        feats = self.encode_images(support_paths, batch_size)
        labels_t = torch.tensor(support_labels).to(self.device)
        protos = []
        for i in range(len(self.names_zh)):
            cf = feats[labels_t == i]
            proto = cf.mean(dim=0) if len(cf) > 0 else torch.zeros(feats.shape[1], device=self.device)
            proto = proto / (proto.norm() + 1e-6)
            protos.append(proto)
        self._prototypes = torch.stack(protos, dim=0)
        return self._prototypes

    # ---------------- 推理 ----------------
    @torch.no_grad()
    def recognize(self, images, topk=5, batch_size=None):
        """
        输入：images (list[path|PIL.Image])
        输出：(pred_idx[list], conf[list], topk_list[list[(idx,name,score)]])
        """
        if self.mode == "few_shot":
            if self._prototypes is None:
                raise RuntimeError("few_shot 模式需先 build_prototypes(support_paths, support_labels)")
            refs = self._prototypes
        else:
            if self._text_feats is None:
                self.build_text_feats()
            refs = self._text_feats

        feats = self.encode_images(images, batch_size)
        logits = 100.0 * feats @ refs.T          # [N, C]
        probs = logits.softmax(dim=-1)            # 概率化置信度
        preds = probs.argmax(dim=-1).tolist()

        topk_list = []
        confs = []
        for i in range(feats.shape[0]):
            sc = probs[i]
            top = sc.topk(min(topk, len(self.names_zh)))
            topk_list.append([(int(idx), self.names_zh[idx], float(s)) for s, idx in zip(top.values.tolist(), top.indices.tolist())])
            confs.append(float(sc[preds[i]]))
        return preds, confs, topk_list


# ---------------- 便捷函数：从 split.csv 收集图 + 标签 ----------------
def gather_from_split(data_dir, split):
    """读 dataset_50cls/<split>.csv，返回 (paths[abs], labels)。"""
    from models.common import ROOT
    csv_path = os.path.join(ROOT, data_dir, f"{split}.csv")
    df = pd.read_csv(csv_path)
    paths = [os.path.join(ROOT, data_dir, str(p).replace("\\", "/")) for p in df["path"]]
    labels = df["label"].tolist()
    return paths, labels


# ---------------- 自测入口 ----------------
if __name__ == "__main__":
    import sys
    print(f"环境：device 检测中...")
    cfg = load_config()
    print(f"数据集: {cfg['project']['data_dir']}  类别数: {cfg['project']['n_classes']}")

    # 默认自测：zero_shot 在 test 上跑 Top-1/Top-5
    mode = sys.argv[1] if len(sys.argv) > 1 else "zero_shot"
    print(f"\n[自测] mode={mode}")
    rec = FoodRecognizer(mode=mode)
    paths, labels = gather_from_split(rec.data_dir, "test")

    # few_shot 需要支持集
    if mode == "few_shot":
        import random
        random.seed(cfg["recognition"]["few_shot"]["seed"])
        k = cfg["recognition"]["few_shot"]["k_shot"]
        sup_paths, sup_labels = [], []
        for idx in rec.class_idx:
            cls_paths = [p for p, l in zip(paths, labels) if l == idx][:k]
            sup_paths += cls_paths
            sup_labels += [idx] * len(cls_paths)
        rec.build_prototypes(sup_paths, sup_labels)

    t0 = time.time()
    preds, confs, topk_list = rec.recognize(paths)
    dt = time.time() - t0
    labels_t = torch.tensor(labels)
    preds_t = torch.tensor(preds)
    top1 = (preds_t == labels_t).float().mean().item()
    # top5
    top5 = sum(labels[i] in [t[0] for t in topk_list[i][:5]] for i in range(len(labels))) / len(labels)
    print(f"\n========== 结果 ==========")
    print(f"mode: {mode}  | test={len(paths)} 张  | 用时 {dt:.1f}s")
    print(f"Top-1: {top1*100:.2f}%  Top-5: {top5*100:.2f}%")
    print(f"平均置信度: {sum(confs)/len(confs):.3f}")
