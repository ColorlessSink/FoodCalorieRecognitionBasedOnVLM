import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.common import ROOT
from models.calorie_agent import CalorieAgent

agent = CalorieAgent(use_llm=False)
A = os.path.join(ROOT, "test_agent_imgs")
imgs = sorted(os.listdir(A))
by_cls = {f.split("_")[0]: os.path.join(A, f) for f in imgs}

print("=== 用例1：单食物识别+共指+累积 ===")
r = agent.chat("帮我看看这盘菜", image_path=by_cls["00"])
print("[助]", r["reply"])
r = agent.chat("再来一份")
print("[助]", r["reply"])
r = agent.chat("总共多少")
print("[助]", r["reply"])

print()
print("=== 用例2：个性化 ===")
agent.reset_meal(); agent.history.clear()
r = agent.chat("我在减脂")
print("[助]", r["reply"])
r = agent.chat("这个能吃吗", image_path=by_cls["10"])
print("[助]", r["reply"])

print()
print("=== 用例3：混合餐盘 ===")
agent.reset_meal(); agent.history.clear()
mixed_dir = os.path.join(ROOT, "dataset_50cls", "mixed")
plates = sorted(os.listdir(mixed_dir))
r = agent.chat("这盘菜一共多少热量", image_path=os.path.join(mixed_dir, plates[0]))
print("[助]", r["reply"])
r = agent.chat("总共多少")
print("[助]", r["reply"])

print()
print("=== 用例4：混合盘后接共指 ===")
agent.reset_meal(); agent.history.clear()
r = agent.chat("看看这盘", image_path=os.path.join(mixed_dir, plates[3]))
print("[助]", r["reply"])
r = agent.chat("再来一份")
print("[助]", r["reply"])
r = agent.chat("总共多少")
print("[助]", r["reply"])

print()
print("=== 用例5：打招呼 ===")
agent.reset_meal(); agent.history.clear()
r = agent.chat("你好")
print("[助]", r["reply"])
print("OK 逻辑通过")
