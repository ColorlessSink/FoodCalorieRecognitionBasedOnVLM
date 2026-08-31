'''
生成实验图表
---
背景：文字数字不直观，图表能一眼看出各方法差距、误差分布、训练曲线。
本脚本读 results/*.json 产出一组 PNG，供报告/演示直接引用。
中文字体用 SimHei（Windows 自带），缺则回退英文标注

产出（写入 results/figures/）：
  fig1_recognition_compare.png  : 识别方法 Top-1/Top-5 对比柱状图
  fig2_portion_methods.png      : 几何法 vs CoT 的 MAE/相对误差
  fig3_ablation_components.png  : 消融A 各组件贡献
  fig4_ablation_geoweight.png   : 消融B geo_weight 扫描曲线
  fig5_lora_train_curve.png     : 50类 LoRA 训练曲线（loss + val_top1）
  fig6_scene_eval.png           : 三场景识别/分量表现
'''
import os, sys, json

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _setup_font():
    # 中文字体；缺字库则关 unicode_minus 避免负号方块
    for f in ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]:
        try:
            from matplotlib.font_manager import FontProperties
            if f in {ff.name for ff in matplotlib.font_manager.fontManager.ttflist}:
                plt.rcParams["font.sans-serif"] = [f]
                plt.rcParams["axes.unicode_minus"] = False
                return f
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _load(name):
    p = os.path.join(ROOT, "results", name)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fig1_recognition():
    data = _load("recognition_baseline.json")
    if not data:
        print("  跳过 fig1：无 recognition_baseline.json")
        return
    labels = [r.get("template") or r["method"] for r in data]
    labels = [l if len(l) < 14 else l[:12] + ".." for l in labels]
    top1 = [r["top1"] for r in data]
    top5 = [r["top5"] for r in data]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, top1, w, label="Top-1", color="#4C72B0")
    ax.bar(x + w/2, top5, w, label="Top-5", color="#55A868")
    ax.axhline(60, ls="--", c="red", lw=1, label="Top-1 阈值 60%")
    ax.axhline(85, ls=":", c="orange", lw=1, label="Top-5 阈值 85%")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("准确率 %"); ax.set_title("食物识别各方法对比（50类 test=600）")
    ax.legend()
    for i, v in enumerate(top1):
        ax.text(i - w/2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig1_recognition_compare.png"), dpi=130)
    plt.close(fig)
    print("  ✓ fig1_recognition_compare.png")


def fig2_portion():
    data = _load("portion_calorie_eval.json")
    if not data:
        print("  跳过 fig2：无 portion_calorie_eval.json")
        return
    methods = list(data.keys())
    mae = [data[m]["mae_g"] for m in methods]
    rel = [data[m]["rel_err_pct"] for m in methods]
    cal = [data[m]["calorie_mae_kcal"] for m in methods]
    x = np.arange(len(methods)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w, mae, w, label="分量 MAE (g)", color="#4C72B0")
    b2 = ax.bar(x, cal, w, label="卡路里 MAE (kcal)", color="#DD8452")
    ax2 = ax.twinx()
    ax2.bar(x + w, rel, w, label="相对误差 (%)", color="#55A868")
    ax2.set_ylabel("相对误差 %"); ax2.set_ylim(0, max(rel)*1.3)
    ax.axhline(30, ls="--", c="red", lw=1, label="分量阈值 30g")
    ax.set_xticks(x); ax.set_xticklabels(["几何法", "CoT法(LLM)"])
    ax.set_ylabel("MAE (g / kcal)"); ax.set_title("分量/卡路里估计方法对比")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig2_portion_methods.png"), dpi=130)
    plt.close(fig)
    print("  ✓ fig2_portion_methods.png")


def fig3_ablation_components():
    data = _load("ablation_study.json")
    if not data or "ablation_A_components" not in data:
        print("  跳过 fig3：无 ablation_study.json")
        return
    a = data["ablation_A_components"]
    keys = ["full", "no_ar", "no_clamp", "only_ar"]
    mae = [a[k]["mae_g"] for k in keys]
    p90 = [a[k]["p90_err_g"] for k in keys]
    x = np.arange(len(keys)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, mae, w, label="MAE (g)", color="#4C72B0")
    ax.bar(x + w/2, p90, w, label="P90 误差 (g)", color="#C44E52")
    ax.axhline(30, ls="--", c="red", lw=1, label="阈值 30g")
    ax.set_xticks(x); ax.set_xticklabels(["完整方法", "去面积比", "去钳制", "去先验锚"])
    ax.set_ylabel("误差 (g)"); ax.set_title("消融A：分量估计各组件的贡献")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig3_ablation_components.png"), dpi=130)
    plt.close(fig)
    print("  ✓ fig3_ablation_components.png")


def fig4_ablation_geoweight():
    data = _load("ablation_study.json")
    if not data or "ablation_B_geo_weight" not in data:
        print("  跳过 fig4：无 ablation_study.json")
        return
    b = data["ablation_B_geo_weight"]
    xs = [float(k.split("=")[1]) for k in b.keys()]
    mae = [b[k]["mae_g"] for k in b.keys()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, mae, "o-", color="#4C72B0", lw=2, markersize=8)
    ax.axhline(30, ls="--", c="red", lw=1, label="阈值 30g")
    ax.axvline(0.2, ls=":", c="green", lw=1, label="当前值 0.2")
    ax.set_xlabel("geo_weight (几何权重)"); ax.set_ylabel("MAE (g)")
    ax.set_title("消融B：geo_weight 扫描")
    ax.legend()
    for x, v in zip(xs, mae):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig4_ablation_geoweight.png"), dpi=130)
    plt.close(fig)
    print("  ✓ fig4_ablation_geoweight.png")


def fig5_lora_curve():
    data = _load("lora_50_test_summary.json")
    if not data or "history" not in data:
        print("  跳过 fig5：无 lora_50_test_summary.json")
        return
    h = data["history"]
    ep = [x["epoch"] for x in h]
    loss = [x["train_loss"] for x in h]
    acc = [x["val_top1"] for x in h]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, loss, "o-", color="#DD8452", lw=2, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("InfoNCE loss", color="#DD8452")
    ax1.tick_params(axis="y", labelcolor="#DD8452")
    ax2 = ax1.twinx()
    ax2.plot(ep, acc, "s-", color="#4C72B0", lw=2, label="val Top-1")
    ax2.set_ylabel("val Top-1 (%)", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.axhline(60, ls="--", c="gray", lw=1)
    plt.title(f"50类 LoRA 训练曲线（test Top-1={data.get('test_top1')}%）")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig5_lora_train_curve.png"), dpi=130)
    plt.close(fig)
    print("  ✓ fig5_lora_train_curve.png")


def fig6_scene():
    data = _load("scene_eval.json")
    if not data:
        print("  跳过 fig6：无 scene_eval.json")
        return
    scenes = list(data.keys())
    top1 = [data[s]["top1"] for s in scenes]
    mae = [data[s]["portion_mae_g"] for s in scenes]
    x = np.arange(len(scenes)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, top1, w, label="识别 Top-1 (%)", color="#4C72B0")
    ax.bar(x + w/2, mae, w, label="分量 MAE (g)", color="#55A868")
    ax.axhline(60, ls="--", c="blue", lw=1)
    ax.axhline(30, ls="--", c="green", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(scenes)
    ax.set_title("跨场景泛化：识别与分量表现")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig6_scene_eval.png"), dpi=130)
    plt.close(fig)
    print("  ✓ fig6_scene_eval.png")


def main():
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    f = _setup_font()
    print(f"字体 {f or '回退英文'}")
    print("==== 生成图表 ====")
    fig1_recognition()
    fig2_portion()
    fig3_ablation_components()
    fig4_ablation_geoweight()
    fig5_lora_curve()
    fig6_scene()
    print("\n完成 results/figures/")


if __name__ == "__main__":
    main()
