'''
models/calorie_agent.py — 智能体交互模块（模块4）
================================================
为什么需要、为什么这么设计：
  前三个模块（识别/分量/营养）是"无状态的算子"，给一张图吐一个 JSON。
  大作业模块4 要求"智能体具备多轮对话、上下文记忆、共指消解、个性化能力"，
  并要求"≥2 个创新点"。本模块把这些"有状态的对话能力"包成一个 Agent：

  职责：
    - 编排 pipeline：识别 → 分量 → 营养（复用模块1/2/3）
    - 多轮上下文记忆：保留最近 max_history 轮，喂给 LLM 做连贯对话
    - 共指消解：把"它/这个/再来一份/刚才那个菜"解析到上下文里最近一次食物
    - 个性化：按用户目标（减脂/增肌/维持/控糖）调整建议口径与宏量分配
    - 一餐累积：跨轮累加已报过的食物，支持"今天总共吃了多少"
    - 追问闭环（v2 新增）：低置信反问后记住 pending 状态，用户答菜名/序号
      即收敛入账；分量纠正（"只吃了一半"）按比例改账，不再答非所问

  创新点（≥2，见 process_log.md §2.4）：
    ① 置信度门控：识别置信度低于阈值时，主动反问"是不是 XX？"而非硬猜，
       把 CLIP 的不确定性显式带进对话，避免低置信错误链式污染下游分量/营养。
    ② 共指+一餐状态机：用"最近食物栈 + 累积宏量"做轻量状态机，让"再来一份/
       它有多少热量/今天总共"这类省略指代能被正确解析，无需 LLM 记忆全历史。
    ③ 个性化目标适配：同一份食物，减脂用户提示"建议分次少油"，增肌用户提示
       "蛋白占比可，碳水中上"，把单一识别结果因人而异地解释。

  LLM 用法：glm-5.2 只做"自然语言生成"（把结构化结果+对话历史组织成回复），
  关键逻辑（共指/累积/门控）用确定性规则做，LLM 失败时规则兜底仍能对话。
'''
import os
import json
import re
import copy
from collections import deque

from models.common import ROOT, load_config


class CalorieAgent:
    def __init__(self, cfg=None, use_llm=True, recognizer=None, portion_estimator=None, nutrition_querier=None):
        self.cfg = cfg or load_config()
        self.use_llm = use_llm and self.cfg["llm"].get("enabled", True)
        self.max_history = self.cfg["agent"].get("max_history", 10)
        self.goals = set(self.cfg["agent"].get("goals", ["减脂", "增肌", "维持", "控糖"]))

        # 三个下游模块：外部可注入（便于测试/复用已加载模型），否则懒加载
        self._recognizer = recognizer
        self._portion = portion_estimator
        self._nutrition = nutrition_querier
        self._detector = None          # MixedDetector 懒加载（混合餐盘）
        self._llm = None

        # ---------- 对话状态 ----------
        self.history = deque(maxlen=self.max_history)   # [{"role","content"}]
        self.last_food = None        # 最近一次识别到的食物 {idx,name,weight,kcal,...}
        self.meal_items = []         # 本餐累积 [{idx,name,weight,kcal,protein,fat,carbs}]
        self.user_goal = None        # 个性化目标
        # 门控追问上下文：低置信反问后，等用户用菜名/序号回答。
        # None = 无待澄清；否则 {"image_path", "topk": [(idx,name,score),...]}。
        # 这是"多轮追问闭环"的关键状态——没有它，反问后的回答会被
        # 误路由到"纠正/闲聊"分支，追问永远不收敛（见 process_log.md agent 改进）。
        self.pending_clarification = None

        # 门控阈值：低于 max(confidence_gate, target_top1×0.5) 时反问。
        # 放进 config（agent.confidence_gate）而非硬编码，消融/调参可配。
        self.conf_gate = float(self.cfg["agent"].get("confidence_gate", 0.3))

    # ---------------- 懒加载下游 ----------------
    @property
    def recognizer(self):
        if self._recognizer is None:
            from models.food_recognizer import FoodRecognizer
            self._recognizer = FoodRecognizer(cfg=self.cfg, mode="zero_shot")
        return self._recognizer

    @property
    def mixed_detector(self):
        if self._detector is None:
            from models.mixed_detector import MixedDetector
            self._detector = MixedDetector(self.cfg)
        return self._detector

    @property
    def portion(self):
        if self._portion is None:
            from models.portion_estimator import PortionEstimator
            self._portion = PortionEstimator(cfg=self.cfg, use_llm=False)
        return self._portion

    @property
    def nutrition(self):
        if self._nutrition is None:
            from models.nutrition_querier import NutritionQuerier
            self._nutrition = NutritionQuerier(cfg=self.cfg)
        return self._nutrition

    @property
    def llm(self):
        if self._llm is None and self.use_llm:
            from models.llm_client import LLMClient
            self._llm = LLMClient(self.cfg)
        return self._llm

    # ============================================================
    #  对话主入口
    # ============================================================
    def chat(self, user_input, image_path=None):
        """
        user_input : 用户文本
        image_path: 可选，本次带图
        返回: {"reply": str, "state": {...}, "turn": int}
        """
        self.history.append({"role": "user", "content": user_input})
        # 门控追问闭环：上一轮 agent 反问"是不是 XX？"且用户还没回答时，
        # 本轮输入先尝试当"澄清回答"解析（菜名/序号/"就是X"）。
        # 命中 → 走收敛分支入账；未命中（如用户换了话题问总量）→ 保持
        # 常规意图路由，pending 状态不误吞新指令。
        if self.pending_clarification and not image_path:
            resolved = self._try_resolve_clarification(user_input)
            if resolved is not None:
                reply = self._apply_clarification(resolved)
                self.history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "turn": len(self.history) // 2 + 1,
                    "state": self._state_snapshot(),
                }

        intent = self._understand(user_input)

        # ---- 意图分发 ----
        # 顺序很关键：带图时优先识别。否则"这是什么"里的"这是"会被误判成"纠正"，
        # 导致带图问"这是什么"反而去查菜名"什么"。带图=用户想看图，纠正只在无图时成立。
        # "总共/一共"是例外：带图问"这盘一共多少"时图尚未入账（本方法还没跑完
        # 识别），直接弹去 ask_total 会答"这餐还没记录"。先识别入账，
        # 下一轮"总共多少"自然汇总。所以 ask_total 只在无图时成立。
        if intent == "set_goal":
            reply = self._handle_set_goal(user_input)
        elif intent == "portion_correct" and not image_path:
            reply = self._handle_portion_correct(user_input)
        elif intent == "again" and not image_path:
            reply = self._handle_again(user_input)
        elif intent == "ask_total" and not image_path:
            reply = self._handle_ask_total()
        elif intent == "reset":
            self.reset_meal()
            reply = "好的，已清空本餐记录，开始新一餐。发张照片吧。"
        elif image_path:
            # 带图：识别优先（"纠正"/"闲聊"在有图时退化为识别）
            reply = self._handle_recognize(image_path, user_input)
        elif intent == "greeting":
            reply = self._greeting()
        elif intent == "correct":
            reply = self._handle_correct(user_input)
        else:
            # 普通闲聊/追问：交给 LLM 用上下文回答
            reply = self._handle_freeform(user_input)

        self.history.append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "turn": len(self.history) // 2 + 1,
            "state": self._state_snapshot(),
        }

    # ============================================================
    #  意图理解（规则优先，稳定可控）
    # ============================================================
    def _understand(self, text):
        t = text.strip()
        # 设目标
        for g in self.goals:
            if g in t and any(k in t for k in ["目标", "在", "我是", "我要", "减脂", "增肌", "控糖", "维持"]):
                return "set_goal"
        if any(k in t for k in ["减脂", "增肌", "控糖", "维持体重"]):
            return "set_goal"
        # 询问总量
        if any(k in t for k in ["总共", "一共", "今天吃了多少", "合计", "累计", "总共多少"]):
            return "ask_total"
        # 分量纠正（半份/一半/剩下）：对最近食物按比例重算
        if any(k in t for k in ["半份", "一半", "半碗", "半盘", "只吃了半", "剩一半", "没吃完", "吃不完"]) \
                or re.search(r"(?:只吃|吃了|剩下?)了?([0-9一二三四五六七八九]{1,2})(?:成|分之[一二三四五六七八九十]+)", t):
            return "portion_correct"
        # 再来一份 / 同一个再来。
        # 注意顺序在 portion_correct 之后："再来半份"里"半份"是分量语义优先。
        if any(k in t for k in ["再来一份", "再来一", "再来", "又吃了一份", "再吃一份", "同样的", "再来同样"]):
            return "again"
        # 修正/纠正（"这是"只在无 pending/无图语境下才像纠正，放最后）
        if any(k in t for k in ["不是", "错了", "其实是", "应该是", "我吃的是"]):
            return "correct"
        if "这是" in t and not self.last_food is None and "什么" not in t:
            return "correct"
        # 新一餐重置
        if any(k in t for k in ["新一餐", "清空", "重新开始", "新一顿"]):
            return "reset"
        # 打招呼
        if any(k in t for k in ["你好", "在吗", "hi", "hello", "你好呀"]):
            return "greeting"
        return "freeform"

    # ============================================================
    #  意图处理
    # ============================================================
    def _handle_recognize(self, image_path, user_input):
        if image_path is None:
            return "请把菜品照片发给我，我来帮你认菜、估分量、算热量。"

        # ---- 混合餐盘分支 ----
        # 为什么：单食物流水线假设"一图一食物"，混合盘会被当一道菜处理。
        # MixedDetector.detect 先提议区域；"唯一大区域且面积占比>0.6"是普通
        # 单食物照片（整图有纹理）的特征 → 退回原单食物路径，其余走混合盘。
        from tools.utils import imread_unicode
        img = imread_unicode(image_path)
        if img is None:
            return f"读不到图片：{image_path}"
        regions = self.mixed_detector.detect(img)
        if not self.mixed_detector.is_single_food(img, regions):
            return self._handle_mixed_plate(img, regions)
        # ---- 单食物路径（原逻辑）----

        preds, confs, topk = self.recognizer.recognize([image_path])
        idx, conf = preds[0], confs[0]
        name = self.recognizer.names_zh[idx]

        # 创新点①：置信度门控
        conf_thr = self.cfg["eval"].get("target", {}).get("top1", 0.6)
        if conf < max(self.conf_gate, conf_thr * 0.5):
            top3 = topk[0][:3]
            self.pending_clarification = {"image_path": image_path, "topk": top3}
            self._clarify_image = image_path
            top3_str = ", ".join(f"{t[1]}({t[2]*100:.0f}%)" for t in top3)
            return (f"这张图我不太确定，最像的前三种是：{top3_str}。"
                    f"能告诉我具体是哪个吗？（回复菜名或序号 1/2/3）"
                    f"或者换个角度拍一张。")

        # 分量 + 营养
        pres = self.portion.estimate_geometric(image_path, idx, name)
        weight = pres["weight_g"]
        nres = self.nutrition.compute(idx, weight)

        self.last_food = {**nres, "confidence": round(conf, 3)}
        self.meal_items.append(copy.deepcopy(self.last_food))

        return self._compose_recognize_reply(nres, conf)

    def _compose_recognize_reply(self, nres, conf):
        name = nres["food_name"]
        w = nres["weight_g"]
        kcal = nres["kcal"]
        goal_tip = self._goal_tip(kcal)
        base = (f"识别到：**{name}**（置信度 {conf*100:.0f}%）。\n"
                f"估算分量约 {w:.0f}g，热量约 {kcal:.0f} kcal，"
                f"蛋白 {nres['protein_g']:.0f}g / 脂肪 {nres['fat_g']:.0f}g / 碳水 {nres['carbs_g']:.0f}g。")
        if goal_tip:
            base += "\n" + goal_tip
        return base

    def _handle_mixed_plate(self, img, regions):
        """混合餐盘：逐区域 crop → 识别 → 几何分量 → 营养，盘级求和。
        分量口径与 experiments/mixed_eval.py 完全一致（区域占盘比例归一 +
        类先验调制），保证对话里报的数与评估报告同源。"""
        import cv2
        from PIL import Image
        from models.mixed_detector import MixedDetector

        h, w = img.shape[:2]
        plate_frac = 0.69          # 白盘面积占比 π·300²/640²（与 mixed_eval 同源）

        items = []
        for ri, region in enumerate(regions):
            crop, _ = MixedDetector.crop(img, region)
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            preds, confs, topk = self.recognizer.recognize([pil])
            idx, conf = preds[0], confs[0]
            name = self.recognizer.names_zh[idx]

            # 分量：与 mixed_eval 相同的"区域占盘比例"几何法
            # （plate_frac_area 是中间量，当前几何调制取 1.0：面积受羽化/检测
            #   阈值影响大，先验锚更稳——与 experiments/mixed_eval.py 同口径）
            ar = region["area"] / float(h * w)
            plate_frac_area = ar / plate_frac
            prior = self.portion._prior(idx, "")
            mod = max(self.portion.clamp_lo, min(self.portion.clamp_hi, plate_frac_area))
            weight = (self.portion.geo_weight * prior * mod
                      + (1 - self.portion.geo_weight) * prior)
            nres = self.nutrition.compute(idx, weight)
            items.append({**nres, "confidence": round(conf, 3)})

        # 记账：整盘各组件入一餐栈（用户"总共多少"时按组件累计）
        for it in items:
            self.meal_items.append(copy.deepcopy(it))
        self.last_food = items[-1] if items else None

        # ---- 组织回复：逐组件 + 盘级汇总 ----
        lines = []
        tot_kcal = tot_p = tot_f = tot_c = tot_w = 0.0
        for it in items:
            tot_kcal += it["kcal"]; tot_p += it["protein_g"]
            tot_f += it["fat_g"]; tot_c += it["carbs_g"]; tot_w += it["weight_g"]
            low = it["confidence"] < 0.6
            tag = "（不太确定）" if low else ""
            lines.append(f"  - {it['food_name']}：约 {it['weight_g']:.0f}g，"
                         f"{it['kcal']:.0f} kcal{tag}")
        reply = (f"这盘有 {len(items)} 样菜：\n" + "\n".join(lines) +
                 f"\n整盘合计：约 {tot_w:.0f}g、{tot_kcal:.0f} kcal，"
                 f"蛋白 {tot_p:.0f}g / 脂肪 {tot_f:.0f}g / 碳水 {tot_c:.0f}g。")
        goal_tip = self._goal_tip(tot_kcal)
        if goal_tip:
            reply += "\n" + goal_tip
        return reply

    def _try_resolve_clarification(self, text):
        """把用户的回答解析成待澄清 top-3 之一（或 50 类里的菜名）。
        返回 (idx, name)；解析不出返回 None。支持：
          ① 序号回答："1" "2" "第三个"
          ② 菜名回答："宫保鸡丁" "是宫保鸡丁" "就是那个宫保鸡丁"
          ③ 候选名模糊命中："宫保" → 宫保鸡丁
        """
        t = text.strip()
        if not t:
            return None
        topk = self.pending_clarification.get("topk", [])
        # ① 序号（"1"/"2"/"3"/"第一个"）。纯数字或"第X个"才算，
        # 防止"再来2份"被当序号吞掉。
        m = re.fullmatch(r"[123]|第[一二三1-3]个?", t)
        if m and topk:
            j = int(t[0]) - 1 if t[0].isdigit() else ["一", "二", "三"].index(t[1])
            return (topk[j][0], topk[j][1])
        # ②/③ 菜名：先精确/模糊匹配 50 类，再看是否命中候选
        for idx, zh in zip(self.recognizer.class_idx, self.recognizer.names_zh):
            if zh in t:
                return (idx, zh)
        for cand in topk:
            if cand[1] in t or t in cand[1]:
                return (cand[0], cand[1])
        return None

    def _apply_clarification(self, resolved):
        """用户澄清后：按选定类走完整"分量→营养→入账"，追问闭环收敛。"""
        self.pending_clarification = None
        idx, name = resolved
        # 优先用反问时的原图重走分量（口径与首轮识别一致）
        src = getattr(self, "_clarify_image", None)
        if src and os.path.exists(src):
            pres = self.portion.estimate_geometric(src, idx, name)
            weight = pres["weight_g"]
        else:
            weight = 220.0
        nres = self.nutrition.compute(idx, weight)
        self.last_food = {**nres, "confidence": 1.0}
        self.meal_items.append(copy.deepcopy(self.last_food))
        return self._compose_recognize_reply(nres, 1.0)

    def _handle_again(self, text):
        """共指：再来一份 → 复用最近食物，但分量可含倍数。"""
        if not self.last_food:
            return "你还没告诉过我你吃了什么呢，先发张照片吧。"
        # 解析倍数："再来两份" "再来3份"。
        # 只在触发词后 4 字内取数字，且"数字+分钟/点/天/月"等时间量词要跳过
        # （"再来3分钟前那个"的 3 是时间不是份数——多轮对话鲁棒性修复，
        #  见 process_log.md agent 改进条目）。
        mult = 1
        for m in re.finditer(r"再来|再吃|又吃", text):
            window = text[m.end():m.end() + 4]
            dm = re.search(r"[两二三四五六七八九]|\d+", window)
            if dm:
                s = dm.group(0)
                after = window[dm.end():dm.end() + 2]
                if re.match(r"分钟|小时|天|个月|点|次后", after):
                    continue          # 时间量词，不是份数
                cn = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                      "六": 6, "七": 7, "八": 8, "九": 9}
                mult = cn[s] if s in cn else int(s)
                break
        f = self.last_food
        w = f["weight_g"] * mult
        nres = self.nutrition.compute(f["food_idx"], w)
        nres["food_name"] = f["food_name"]
        self.meal_items.append(copy.deepcopy(nres))
        self.last_food = {**nres, "confidence": f.get("confidence", 0)}
        return (f"好，再记一份 {f['food_name']}×{mult}：{w:.0f}g，"
                f"热量 {nres['kcal']:.0f} kcal。")

    def _handle_portion_correct(self, text):
        """分量纠正：'半份/一半/剩一半' → 对最近食物按比例重算并改账。
        与菜名纠正（_handle_correct）互补：那个改'是什么'，这个改'吃了多少'。"""
        if not self.last_food:
            return "目前还没有记录，先发张照片，再告诉我要减多少。"
        f = self.last_food
        # 比例解析：默认 0.5；"吃了三成/剩七成"按成数折算
        ratio = 0.5
        m = re.search(r"(?:吃了?|剩下?|只吃)([0-9一二三四五六七八九十]{1,2})(?:成|成分)", text)
        if m:
            s = m.group(1)
            cn = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
            ratio = (cn[s] if s in cn else int(s)) / 10.0
        elif "大半" in text:
            ratio = 0.7
        elif "小半" in text:
            ratio = 0.3
        # 同源条目定位：meal_items 里与 last_food 同名且最新的那条
        # （last_food 与 meal_items[-1] 可能不同源——中间夹过"再来一份"）
        tgt = None
        for i in range(len(self.meal_items) - 1, -1, -1):
            if self.meal_items[i]["food_name"] == f["food_name"]:
                tgt = i
                break
        if tgt is None:
            tgt = len(self.meal_items) - 1 if self.meal_items else None
        w = f["weight_g"] * ratio
        nres = self.nutrition.compute(f["food_idx"], w)
        nres["food_name"] = f["food_name"]
        if tgt is not None:
            self.meal_items[tgt] = {**nres, "confidence": f.get("confidence", 0)}
        self.last_food = {**nres, "confidence": f.get("confidence", 0)}
        return (f"好的，按 {ratio*100:.0f}% 重算：{f['food_name']} {w:.0f}g，"
                f"热量约 {nres['kcal']:.0f} kcal。")

    def _handle_correct(self, text):
        """纠正：从文本里抠菜名，查库改 last_food 的类别。"""
        # 去掉前缀，取可能的菜名
        cleaned = re.sub(r"^(不是|错了|其实是|应该是|我吃的是|这是)[，, ]*", "", text)
        # 在类别表里模糊匹配
        target = None
        for idx, zh in zip(self.recognizer.class_idx, self.recognizer.names_zh):
            if zh in cleaned or cleaned in zh:
                target = (idx, zh); break
        if target is None:
            # 交给 LLM 在 50 类里找最像的
            target = self._llm_match_class(cleaned)
        if target is None:
            return f"没在 50 类里找到『{cleaned}』，能再说一次吗？"
        idx, zh = target
        # 沿用上次的分量
        w = self.last_food["weight_g"] if self.last_food else 220.0
        nres = self.nutrition.compute(idx, w)
        # 修正最近一条 meal_items——但必须与 last_food 同源（菜名相同）：
        # 中间夹过"再来一份"时 meal_items[-1] 是另一道菜，盲改会改错账。
        tgt = len(self.meal_items) - 1 if self.meal_items else None
        if self.meal_items and self.last_food:
            for i in range(len(self.meal_items) - 1, -1, -1):
                if self.meal_items[i]["food_name"] == self.last_food["food_name"]:
                    tgt = i
                    break
        if tgt is not None:
            self.meal_items[tgt] = {**nres, "confidence": 0.6}
        self.last_food = {**nres, "confidence": 0.6}
        return (f"好的，记成 **{zh}**，{w:.0f}g，热量约 {nres['kcal']:.0f} kcal。")

    def _handle_set_goal(self, text):
        for g in ["减脂", "增肌", "控糖", "维持"]:
            if g in text:
                self.user_goal = g
                break
        tips = {
            "减脂": "减脂期建议每餐热量控制在 400~550 kcal，蛋白优先、少油少糖。",
            "增肌": "增肌期可适当提高碳水与蛋白，每餐 600~800 kcal，训练后补充。",
            "控糖": "控糖期减少精制碳水（米饭/面条/甜点），多叶菜与蛋白。",
            "维持": "维持期保持热量均衡，注意三大宏量比例约 3:3:4。",
        }
        return f"已记录你的目标：**{self.user_goal}**。{tips.get(self.user_goal, '')}"

    def _handle_ask_total(self):
        if not self.meal_items:
            return "这餐还没记录呢，发张照片开始吧。"
        tot = sum(f["kcal"] for f in self.meal_items)
        tot_p = sum(f["protein_g"] for f in self.meal_items)
        tot_f = sum(f["fat_g"] for f in self.meal_items)
        tot_c = sum(f["carbs_g"] for f in self.meal_items)
        names = "、".join(f"{f['food_name']}({f['weight_g']:.0f}g)" for f in self.meal_items)
        msg = (f"这餐目前记录：{names}。\n"
               f"合计 {tot:.0f} kcal，蛋白 {tot_p:.0f}g / 脂肪 {tot_f:.0f}g / 碳水 {tot_c:.0f}g。")
        if self.user_goal:
            msg += "\n" + self._goal_tip(tot)
        return msg

    def _handle_freeform(self, text):
        if not self.use_llm:
            return self._rule_freeform(text)
        sys_prompt = self._build_system_prompt()
        msgs = [{"role": "system", "content": sys_prompt}] + list(self.history)
        ans = self.llm.chat(msgs, max_tokens=768, fallback=self._rule_freeform(text))
        return ans

    def _rule_freeform(self, text):
        if self.last_food:
            f = self.last_food
            # 追问宏量：从最近食物的结构化结果里取数，避免 LLM 兜底时"答非所问"
            if "蛋白" in text:
                return f"{f['food_name']} 约 {f['weight_g']:.0f}g，蛋白质约 {f['protein_g']:.0f}g。"
            if "脂肪" in text:
                return f"{f['food_name']} 约 {f['weight_g']:.0f}g，脂肪约 {f['fat_g']:.0f}g。"
            if "碳水" in text:
                return f"{f['food_name']} 约 {f['weight_g']:.0f}g，碳水约 {f['carbs_g']:.0f}g。"
            if "热量" in text or "卡" in text:
                return f"{f['food_name']} 约 {f['weight_g']:.0f}g，热量约 {f['kcal']:.0f} kcal。"
            return (f"根据记录，最近一份是 {f['food_name']}，约 {f['weight_g']:.0f}g、"
                    f"{f['kcal']:.0f} kcal。还有别的要问吗？")
        return "把菜品照片发给我，我帮你认菜、估分量、算热量。"

    def _greeting(self):
        return ("你好！我是卡路里识别助手。把菜品照片发给我，"
                "我会识别菜名、估算分量、计算热量和三大宏量。"
                "你可以告诉我你的目标（减脂/增肌/控糖/维持），我会给个性化建议。")

    # ============================================================
    #  个性化
    # ============================================================
    def _goal_tip(self, kcal):
        if not self.user_goal:
            return ""
        if self.user_goal == "减脂":
            if kcal > 600:
                return f"⚠️ 减脂期这一份 {kcal:.0f} kcal 偏高，建议分两餐或减少主食。"
            return "✅ 热量在减脂区间，可以。"
        if self.user_goal == "增肌":
            if kcal < 400:
                return f"增肌期这一份 {kcal:.0f} kcal 偏少，建议搭配主食补充碳水。"
            return "✅ 热量适合增肌期。"
        if self.user_goal == "控糖":
            return "控糖期重点关注碳水，尽量减少米饭/面条/甜点。"
        return "✅ 热量在维持区间。"

    # ============================================================
    #  LLM 辅助
    # ============================================================
    def _build_system_prompt(self):
        ctx = []
        if self.last_food:
            f = self.last_food
            ctx.append(f"最近识别食物：{f['food_name']}，{f['weight_g']:.0f}g，{f['kcal']:.0f} kcal。")
        if self.meal_items:
            tot = sum(x["kcal"] for x in self.meal_items)
            ctx.append(f"本餐已累积 {len(self.meal_items)} 项，共 {tot:.0f} kcal。")
        if self.user_goal:
            ctx.append(f"用户目标：{self.user_goal}。")
        ctx_str = "\n".join(ctx) if ctx else "（暂无上下文）"
        return ("你是一个食物卡路里识别助手。根据已识别的结构化结果与对话上下文，"
                "用简洁中文回答用户。不要编造未给出的数值，不确定时引导用户发照片。"
                f"\n\n当前上下文：\n{ctx_str}")

    def _llm_match_class(self, text):
        """让 LLM 把用户口语菜名映射到 50 类之一。失败返回 None。"""
        if not self.use_llm:
            return None
        names = self.recognizer.names_zh
        prompt = ("从下列 50 个中餐菜名里，选出与用户输入最匹配的一个，只输出菜名本身，"
                  f"不要其它字。\n候选：{', '.join(names)}\n用户输入：{text}")
        ans = self.llm.ask(prompt, max_tokens=64, fallback="")
        ans = ans.strip().strip("。.,，")
        for idx, zh in zip(self.recognizer.class_idx, self.recognizer.names_zh):
            if zh == ans or zh in ans or ans in zh:
                return (idx, zh)
        return None

    # ============================================================
    #  状态快照（给 demo/测试用）
    # ============================================================
    def _state_snapshot(self):
        return {
            "last_food": self.last_food["food_name"] if self.last_food else None,
            "meal_items": len(self.meal_items),
            "meal_kcal": round(sum(f["kcal"] for f in self.meal_items), 1),
            "goal": self.user_goal,
            "history_turns": len(self.history),
            "pending_clarification": bool(self.pending_clarification),
        }

    def reset_meal(self):
        """开始新一餐：清空一餐记录与共指栈，但保留个性化目标
        （用户说"新一餐"是想换一道菜继续吃，不是想换减肥目标）。"""
        self.meal_items = []
        self.last_food = None
        self.pending_clarification = None
