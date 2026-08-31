# 基于 VLM 的食物卡路里识别智能体

> 本仓库为 EagleLab 大作业：基于 VLM 的食物卡路里识别智能体（50 类 + 分量 + 卡路里 + 多轮对话智能体）。识别模块的前身是 20 类中餐 Chinese-CLIP 分类实验（小作业，已归档移除），本仓库只保留大作业完整交付。

## 大作业概述

基于视觉-语言模型（Chinese-CLIP）构建一个食物卡路里识别智能体，完成"拍照 → 认菜 → 估分量 → 算热量 → 多轮对话"全流程。在"中餐数据集无分量标注 + 不使用视觉 LLM"双重约束下，采用**本地视觉 + 远程纯文本 LLM + 规则兜底**的三层解耦架构。

四个模块：①食物识别（Chinese-CLIP 零样本/少样本/LoRA 三模式）②分量估计（GrabCut 分割 + 面积比相对调制 + 类先验锚）③营养查询与卡路里计算 ④多轮对话智能体（置信度门控追问闭环 / 共指一餐状态机 / 半份纠正 / 个性化目标适配）。

## 大作业结果汇总（50 类，test 600 张）

| 指标 | 硬指标 | 本作业 | 达标 |
|------|--------|--------|------|
| 识别 Top-1 | ≥60% | 78.33%(零样本) / 81.50%(LoRA) / 83.67%(少样本)；跨模型 VLM 基线 Qwen2-VL-2B 生成式零样本 23.83%（见下） | 达标 |
| 识别 Top-5 | ≥85% | 97.00% / 98.17% / 98.00%；Qwen2-VL-2B 29.67% | 达标 |
| 识别类别数 | ≥50 | 50 | 达标 |
| 分量 MAE | ≤30g 或 rel≤25% | 全量 24.0-27.2g（人工三场景口径）/ rel 16.7% | 达标 |
| 卡路里 MAE（单） | ≤50kcal | 全量约 45kcal | 达标 |
| 混合餐盘卡路里 MAE | ≤100kcal | oracle 74.0 / e2e 107.8（中位 79.2；v1 120.9→v2 110.9→v3 107.8，v3 为随机化数据口径），组件识别 few_shot 20-shot + TTA + 原型域增强 + 门控 soft-kcal Top-1 83.7%，检测召回 100%（区域数 119/120） | oracle 达标，e2e 见报告分层分析 |
| 智能体用例 | ≥10 | 18 用例 50 轮（含 3 个混合餐盘用例） | 达标 |
| 智能体准确率 | ≥80% | 100%（50/50） | 达标 |
| 创新点 | ≥2 | 3 | 达标 |
| 消融 | ≥2 | 3 | 达标 |
| 失败案例 | ≥15 | 18 | 达标 |
| 文献综述 | ≥3000字≥20引 | 6019字/40引（近3年50%+，顶会顶刊9篇） | 达标 |
| 总结报告 | ≥8000字 | report/final_report.md（12622 汉字） | 达标 |

**跨模型 VLM 基线（"≥2 种 VLM 基线"要求）**：服务器（RTX 3090 24GB，bf16）部署 `Qwen/Qwen2-VL-2B-Instruct`，在相同条件下做生成式零样本分类（`experiments/vlm_baseline_eval.py`）：Top-1 **23.83%** / Top-5 **29.67%**（耗时 3022.9s），对比 Chinese-CLIP 判别式零样本 78.33%/97.00%。说明50 类闭集中餐分类上，域内判别式 CLIP 显著优于 2B 通用生成式 VLM，反向支持本项目技术路线。

**外部网图域外检验**（`dataset_external/` 500 张真实网图 + 40 张真实混合菜，见 `results/external_eval.json`）：单食物零样本 Top-1 **89.80%**（内部 test 78.33%），分量 MAE 23.8g、卡路里 oracle 44.4 kcal，与内部同口径一致；真实混合菜暴露纹理检测器域差（37/40 整图当一块），已换 **SAM ViT-B 后端**（`models/sam_detector.py`）重标：40 张里 37 张拆出 2~8 个组件（`dataset_external/mixed_labels.csv`，纹理后端备份于 `mixed_labels_texture.csv`）。详见 `report/process_log.md §3.9/§3.10` 与 `report/final_report.md §4.3.6`。

**混合盘检测后端对比（SAM vs 纹理 CV，`results/sam_vs_texture.json` + `results/figures/fig7_sam_vs_texture.png`）**：120 合成盘 GT 同口径下，纹理后端区域数 119/120、召回 100%（合成域近满分），SAM 后端区域数 65/120、召回 87.0%、质心偏差 13.4px（聚合四步：面积/整图层过滤 + bbox 稠密度过滤 + IoA-NMS + 质心合并，参数在合成 GT 上扫参选定）。两后端定位互补：**纹理后端合成域强、真实照片塌成整块；SAM 后端合成域居中、但能拆真实照片**——真实部署应选 SAM。SAM 是"分割大模型"（本地零样本掩码提议）非视觉 LLM，与"不用 LLM 视觉能力"约束不冲突。

## 大作业环境

代码在**两套环境**实测通过（`requirements.txt` 顶部注释给出两套环境的完整描述）：

**环境 A（实验主环境）**：本地 Windows 11 + Python 3.13 + RTX 4050 Laptop 6GB

- torch 2.13.0+cu130、transformers 5.14.1、peft 0.19.1、accelerate、datasets
- opencv-python、scikit-learn、matplotlib、pandas、Pillow
- requests、pyyaml（LLM 客户端与配置读取）、gradio 6.26（Web Demo）

**环境 B（复现环境）**：远程 Ubuntu 22.04 + Python 3.10 + RTX 3090 24GB
- torch 2.5.1+cu121、transformers 4.46.3、peft 0.20.0（关键包版本不同，代码已做三路兼容）
- 另部署 Qwen2-VL-2B-Instruct 作跨模型 VLM 基线（`~/vlm_baseline/`，模型经 `HF_ENDPOINT=https://hf-mirror.com` 离线下载，transformers 4.46.3 原生支持）

```bash
pip install -r requirements.txt
```

Chinese-CLIP 模型 `OFA-Sys/chinese-clip-vit-base-patch16` 可提前下载到本地 HuggingFace 缓存，或修改各脚本中的 `_LOCAL_SNAP` 路径（或留空）。

**LLM 配置**：项目本身开发时使用 GLM5.2，不使用 LLM 视觉能力。实际运行时需使用环境变量配置自己的 LLM 接口：

- `ANTHROPIC_BASE_URL=`
- `ANTHROPIC_MODEL=`
- `ANTHROPIC_AUTH_TOKEN=<你的 token>`

所有 LLM 调用带规则兜底（`fallback_to_rules: true`），断网/失败时退化为规则计算，流水线不依赖网络。

## 大作业目录结构

```
.
├── config/config.yaml              配置（模型路径、超参、LLM、识别/分量/智能体参数）
├── data/
│   ├── build_dataset_50.py         构建 50 类数据集 → dataset_50cls/
│   ├── build_nutrition_db.py       营养库（每 100g 热量/三大宏量/密度）
│   ├── build_labels.py             合成分量/卡路里真值（文件名 hash + 类先验高斯，与估计器解耦）
│   ├── build_scene_split.py        场景分层（standard/real/challenge）
│   ├── build_area_stats.py         各桶 area_ratio 中位数（分量调制用）
│   ├── nutrition_db.csv            50 类营养库
│   ├── area_ratio_stats.csv        分量标定表
│   ├── build_mixed_plates.py       合成混合餐盘（→ dataset_50cls/mixed/）
│   ├── build_external_labels.py    外部抓取单食物合成真值（→ dataset_external/single_labels_ext.csv）
│   └── preprocess.py               一键串跑以上全部（含完整性校验）
├── models/
│   ├── food_recognizer.py          模块1：食物识别（零样本/少样本/LoRA）
│   ├── portion_estimator.py        模块2：分量估计（几何法 v2 相对调制 + CoT 法）
│   ├── nutrition_querier.py        模块3：营养查询与卡路里计算
│   ├── calorie_agent.py            模块4：智能体主控（门控/共指/个性化/混合盘）
│   ├── mixed_detector.py           混合餐盘多食物区域检测（纹理线索 + 分水岭拆分；合成盘后端）
│   ├── sam_detector.py             混合餐盘区域检测·SAM ViT-B 后端（真实照片可拆；同接口可替换）
│   ├── llm_client.py               glm-5.2 客户端（OpenAI 兼容，规则兜底）
│   └── common.py                   公共工具（ROOT/config/classes 加载）
├── experiments/
│   ├── baseline_eval.py            识别基线对比
│   ├── mixed_eval.py               混合餐盘分层评估（检测/识别/盘级卡路里）
│   ├── ablation_study.py           消融 A/B/C
│   ├── failure_analysis.py         失败案例捞取
│   ├── train_lora_50.py            50 类 LoRA 训练
│   ├── external_eval.py            外部网图评估（识别/分量/卡路里/混合盘）
│   ├── vlm_baseline_eval.py        跨模型 VLM 基线（Qwen2-VL-2B 生成式零样本，服务器运行）
│   └── visualization.py            可视化（fig1-7）
├── demo/
│   ├── cli_demo.py                 命令行交互 Demo
│   ├── web_demo.py                 Gradio Web Demo（上传图片 + 对话 + 状态面板）
│   └── dialogue_cases.json         18 个对话测试用例（含 3 个混合餐盘）
├── tools/
│   ├── label_scene.py              三场景人工标注工具（看图按键，修正自动划分）
│   ├── scrape_images.py            网图抓取（百度/Bing，500 单食物 + 40 真实混合菜）
│   ├── label_mixed.py              真实混合菜机器预标（检测 → 裁剪 → few-shot 预标 → 复核大图）
│   ├── inspect_model.py            查看 Chinese-CLIP 结构、定位 LoRA Q/V 注入层
│   └── utils.py                    通用工具
├── report/
│   ├── literature_review.md        文献综述（6019字/40引）
│   ├── process_log.md              过程日志（含难点与错误记录）
│   ├── failure_cases.md            失败案例（18 例）
│   ├── final_report.md             总结报告（≥8000字）
│   └── TODO_USER.md                用户待完成清单（P0/P1/P2，含答辩与补测指引）
├── literature/                     文献综述参考文献 PDF（已下载部分，其余因出版方付费墙暂无法获取）
├── dataset_50cls/                  50 类数据集（train1500/val300/test600 + mixed/ 120 盘 + 真值 + 场景分层）
├── dataset_external/               外部抓取数据（images/ 500 单食物 + mixed_raw/ 40 真实混合菜 + 真值 + 机器预标）
└── results/                        实验产出（JSON + figures/）
```

## 大作业 Quick Start

所有脚本在**项目根目录**运行。

#### 1. 构建数据集（50 类 + 营养库 + 真值 + 场景分层 + 混合盘）

确保 `ChineseFood Net 3/release_data/` 原始数据存在：

```bash
python data/build_dataset_50.py    # 50 类数据集 → dataset_50cls/
python data/build_nutrition_db.py # 营养库 → data/nutrition_db.csv
python data/build_labels.py       # 合成真值 → dataset_50cls/test_labels.csv
python data/build_area_stats.py   # 面积比标定 → data/area_ratio_stats.csv
python data/build_scene_split.py  # 场景分层（自动统计代理）→ dataset_50cls/test_scene.csv
python data/build_mixed_plates.py # 混合餐盘 120 盘 → dataset_50cls/mixed/
```

以上 6 步也可一键跑：`python data/preprocess.py`（顺序相同，末尾附完整性校验）。
**注意**：首步会清空重建 `dataset_50cls/`，已有的场景标注进度会丢失，重跑前先备份 `scene_labels_progress.json` 与 `test_scene.csv`。

#### 1b. 三场景人工标注（可选，用于修正自动划分）

```bash
python tools/label_scene.py       # 看图按键标注，随时可中断续标
```

输出与 `build_scene_split.py` 完全对齐；进度实时写 `dataset_50cls/scene_labels_progress.json`；首次导出前自动把自动版备份为 `test_scene_auto.csv`（之后不覆盖备份）；未标满 600 张会拒绝导出。

#### 2. 训练 50 类 LoRA（可选，零样本/少样本已达标）

```bash
python experiments/train_lora_50.py    # adapter → results/lora_adapter_50/，约 6 分钟
```

#### 3. 评估

```bash
python experiments/baseline_eval.py    # 识别基线 → results/recognition_baseline.json
python experiments/mixed_eval.py       # 混合餐盘分层评估 → results/mixed_plate_eval.json
python experiments/ablation_study.py   # 消融 A/B/C → results/ablation_study.json
python experiments/failure_analysis.py # 失败案例 → results/failure_cases_raw.json
python experiments/visualization.py    # 图表 → results/figures/fig1-7.png
python -c "from models.sam_detector import SamDetector"   # SAM 后端自检（可选）
```

#### 3b. 外部网图评估（可选，域外泛化检验）

```bash
python data/build_external_labels.py  # 外部单食物合成真值 → dataset_external/single_labels_ext.csv
python experiments/external_eval.py   # 识别（三模式）+ 分量/卡路里 + 真实混合盘统计 → results/external_eval.json
```

`tools/scrape_images.py` 抓的 500 张单食物（50 类 × 10）+ 40 张真实混合菜存在 `dataset_external/`，与 `dataset_50cls/` 完全隔离、不破坏原流程。

#### 3c. 跨模型 VLM 基线（可选，"≥2 种 VLM 基线"要求；需 GPU 服务器）

```bash
# 服务器上：把 test.csv / classes_50.csv / test_imgs/ 与脚本放到同一目录（项目里为 ~/vlm_baseline/）
# 模型经 HF_ENDPOINT=https://hf-mirror.com 下载 Qwen2-VL-2B-Instruct 后离线运行
python3 vlm_baseline_eval.py    # → vlm_baseline_result.json（拷回本地并入 results/recognition_baseline.json）
```

在 RTX 3090 24GB（bf16）上 600 张约 50 分钟。与 Chinese-CLIP 判别式基线同数据集同标签口径（生成式 Top-5 为模型自报备选，语义与判别式略有差异，结果 JSON 中有声明）。

#### 4. 智能体交互（参照上方 LLM 配置使用环境变量设置 LLM 接口）

```bash
# 单图识别
python demo/cli_demo.py --image dataset_50cls/test/00_麻婆豆腐/00_003720.jpg

# 跑 18 个对话用例（--no-llm 用规则兜底，默认调 glm-5.2）
python demo/cli_demo.py --script demo/dialogue_cases.json --no-llm
```

#### 5. Web Demo（Gradio）（参照上方 LLM 配置使用环境变量设置 LLM 接口）

```bash
python demo/web_demo.py             # 浏览器打开 http://127.0.0.1:7860
python demo/web_demo.py --no-llm    # 禁用 LLM，纯规则模式
python demo/web_demo.py --port 7863 # 指定端口（多用户共用机器时）
```

上传单食物或混合餐盘照片 + 文字提问，右侧状态面板实时显示本餐累积与个性化目标；"新一餐"按钮清空记录。左列为对话主区，下接统一输入区`[ 菜品照片 | 文字 + 发送/新一餐 ]`，按工作流组织（图片是主要输入占左格、文字补充在右、发送与新一餐同为"动作"归按钮行）。

## 大作业关键说明

- **合成真值诚实声明**：ChineseFoodNet 无分量/卡路里标注，真值按"文件名 md5 hash + 类先验高斯"采样，**与估计器输入（图像分割区域）完全解耦**，保证评估公平。详见 `report/process_log.md §0.4` 与 `report/final_report.md §3.4`。
- **三层解耦架构**：开发时使用 glm-5.2 ，不使用大模型视觉能力，故视觉理解由本地 Chinese-CLIP 承担，LLM 只做纯文本 CoT 与自然语言生成，关键逻辑规则兜底。详见 `report/process_log.md §0.3`。
- **分量估计 v2**：v1 绝对几何标定（MAE 143.1g）失败，于是采用 v2 相对调制 + 类先验锚（全量 24-25g，达标）。根因是中餐图片无统一参考物、绝对标定被分辨率系统性污染。详见 `report/process_log.md §2.2`。

---

## 关键实现说明

- **Chinese-CLIP 特征坑**：`get_text_features`/`get_image_features` 返回的不是最终对齐嵌入，要手动取 `last_hidden_state[:,0,:]`（CLS）再乘 `text_projection`/`visual_projection`，否则相似度全乱。
- **LoRA 注入位置**：文本塔（`query`/`value`）和视觉塔（`q_proj`/`v_proj`）命名不一致，详见 `tools/inspect_model.py`。
- **InfoNCE 损失**：CLIP 式双向交叉熵，对角线为正样本，batch 内其余为负样本。
- **过拟合防止**：每 epoch 在 val 集评估，只保存 val 最优的 adapter。

## 备注

- `ChineseFood Net 3/`（原始数据）、`dataset_50cls/`（构建产物，2400+ 张图，含 `test_scene*.csv` 与标注进度 `scene_labels_progress.json`）、`.claude/` 已在 `.gitignore` 中忽略，不入库；**`results/`（含 figures）会入库**，它们是报告数字的凭证。
- 各脚本中的 `_LOCAL_SNAP` 是本机 HuggingFace 缓存快照路径，换机器运行时需改为自己的路径或留空使用在线模型名。
