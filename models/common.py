'''
models/common.py — 通用配置与模型加载工具
================================================
为什么需要这个文件：
  小作业里每个脚本都各自硬编码 _LOCAL_SNAP、device、DATA_DIR，复制粘贴 6+ 份，
  改一处要改多处。大作业模块更多，把"读 config.yaml + 加载 Chinese-CLIP + 取类别表"
  收敛到一个公共模块，所有 models/ 文件 import 它，保证口径一致。

关键设计：
  - load_config() 用绝对路径定位 config/config.yaml，不管从哪个目录运行都找得到。
  - load_clip() 返回 (model, processor)，并复用小作业验证过的"本地快照优先"策略，
    避免联网校验超时。返回前 .eval() 且不 .to(device)——device 由调用方决定，
    这样同一份加载结果既能用于推理也能用于 LoRA 训练。
'''
import os
import yaml
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")


def load_config():
    """读取 config/config.yaml，返回 dict。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clip_model_name(cfg):
    """优先用本地快照路径，否则用 hub id（小作业踩坑：联网校验会超时）。"""
    m = cfg["model"]["clip"]
    snap = m.get("local_snapshot")
    return snap if snap else m["name"]


def load_clip(cfg=None, device=None):
    """
    加载 Chinese-CLIP base 模型与 processor。
    返回 (model, processor)。model 不 .to(device)，由调用方决定（便于训练/推理复用）。
    """
    if cfg is None:
        cfg = load_config()
    name = clip_model_name(cfg)
    model = ChineseCLIPModel.from_pretrained(name, low_cpu_mem_usage=True)
    processor = ChineseCLIPProcessor.from_pretrained(name)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model, processor


def load_classes(data_dir):
    """读 dataset_50cls/classes_50.csv（或 20cls 的 classes.csv），返回 (names_zh, idxs)。"""
    # 50 类叫 classes_50.csv，20 类叫 classes.csv
    for fn in ("classes_50.csv", "classes.csv"):
        p = os.path.join(ROOT, data_dir, fn)
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df["zh"].tolist(), df["idx"].tolist(), df
    raise FileNotFoundError(f"找不到类别表 in {data_dir}")


def device_of(cfg):
    return cfg["model"]["device"] if cfg and "model" in cfg else ("cuda" if torch.cuda.is_available() else "cpu")
