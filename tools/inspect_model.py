'''
脚本：查看ChineseCLIPModel的模型结构，找出其中的Q(query)和V(value)层名字
---
LoRA微调任务的前置脚本
'''

import torch.nn as nn
from transformers import ChineseCLIPModel


MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
_LOCAL_SNAP = r"C:\Users\Bill\.cache\huggingface\hub\models--OFA-Sys--chinese-clip-vit-base-patch16\snapshots\f4a64596bbcf9a2a94591b74b9dc39b2e4e77e3e"
MODEL_NAME = _LOCAL_SNAP if _LOCAL_SNAP else MODEL_NAME  # 本地缓存快照（避免每次联网校验导致超时）

device = "cuda"

model = ChineseCLIPModel.from_pretrained(MODEL_NAME).to(device)

for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        print(name, mod.weight.shape)
