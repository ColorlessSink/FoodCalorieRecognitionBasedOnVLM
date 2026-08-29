'''
demo/cli_demo.py — 智能体命令行演示
================================================
为什么需要：
  大作业要求"提供完整代码与运行示例"。这里提供一个可交互的 CLI，
  直接驱动 CalorieAgent：发图/发文本，看多轮对话、共指、个性化、一餐累积。

用法：
  python -m demo.cli_demo                  # 交互模式
  python -m demo.cli_demo --image path.jpg # 带图启动一轮
  python -m demo.cli_demo --script cases/dialogue_cases.json  # 跑测试用例
'''
import os
import sys
import json
import argparse

# 以 `python demo/cli_demo.py` 方式启动时，Python 只把 demo/ 加进 sys.path，
# 找不到根目录下的 models 包。这里把项目根目录（demo 的上一级）补进搜索路径，
# 使 `python demo/cli_demo.py` 与 `python -m demo.cli_demo` 两种方式都能运行。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.stdout.reconfigure(encoding="utf-8")

from models.common import ROOT
from models.calorie_agent import CalorieAgent


def banner():
    print("=" * 60)
    print("  卡路里识别智能体 · CLI 演示")
    print("  发图：  img <图片路径>")
    print("  发文字：直接输入")
    print("  退出：  quit")
    print("=" * 60)


def run_interactive(agent, first_image=None):
    banner()
    if first_image:
        r = agent.chat("帮我看看这盘菜", image_path=first_image)
        print(f"[助手] {r['reply']}\n")
    while True:
        try:
            line = input("[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if line.lower().startswith("img "):
            img = line[4:].strip().strip('"')
            if not os.path.isabs(img):
                img = os.path.join(ROOT, img)
            if not os.path.exists(img):
                print(f"  找不到图片：{img}")
                continue
            r = agent.chat("帮我看看这盘菜", image_path=img)
        else:
            r = agent.chat(line)
        print(f"[助手] {r['reply']}")
        st = r["state"]
        print(f"  (本餐 {st['meal_items']} 项 / {st['meal_kcal']} kcal / 目标 {st['goal'] or '未设'})\n")


def run_script(agent, script_path):
    """跑一组对话测试用例，输出每轮结果。"""
    with open(script_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    print(f"运行 {len(cases)} 个对话用例...\n")
    results = []
    for i, c in enumerate(cases, 1):
        # 用例间完全隔离：清空一餐记录、对话历史、个性化目标。
        # 否则上一个用例设的"控糖/减脂"会泄漏到后续用例，污染断言。
        agent.reset_meal()
        agent.history.clear()
        agent.user_goal = None
        print(f"--- 用例 {i}：{c.get('name','')} ---")
        conv = []
        for turn in c["turns"]:
            img = turn.get("image")
            if img and not os.path.isabs(img):
                img = os.path.join(ROOT, img)
            text = turn["text"]
            r = agent.chat(text, image_path=img)
            print(f"  [你] {text}" + (f"  (img:{turn.get('image')})" if img else ""))
            print(f"  [助] {r['reply']}\n")
            conv.append({"text": text, "image": turn.get("image"),
                         "reply": r["reply"], "state": r["state"]})
        results.append({"name": c.get("name", ""), "turns": conv})
    out = os.path.join(ROOT, "results", "dialogue_test_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="启动时先发一张图")
    parser.add_argument("--script", default=None, help="跑对话测试用例 JSON")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM，纯规则")
    args = parser.parse_args()

    agent = CalorieAgent(use_llm=not args.no_llm)
    if args.script:
        run_script(agent, args.script)
    else:
        run_interactive(agent, first_image=args.image)
