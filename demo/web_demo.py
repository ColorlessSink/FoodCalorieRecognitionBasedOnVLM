'''
demo/web_demo.py — Web 界面 Demo（Gradio 6.x）
================================================
为什么需要：
  大作业第五部分代码结构要求 demo/web_demo.py（Gradio/Streamlit Web 界面），
  "交互方式可选：命令行交互、Web界面（Gradio/Streamlit）、或API服务"。
  Web 版比 CLI 多出"上传图片 + 对话气泡 + 状态面板"的真实产品形态，
  也是答辩演示最直观的入口。

设计：
  - 复用 CalorieAgent（与 CLI 完全同一套逻辑：门控/共指/累积/混合盘），
    Web 层只做 IO（Gradio 临时文件路径 → agent.chat → 展示）。
  - 全局单个 agent 实例（Gradio Session State 存复杂对象需 reload 兜底，
    单实例最稳）；配合"新一餐"按钮清空状态。多用户会共享上下文——
    单机演示场景可接受，代码注释里写明限制。
  - 状态面板每轮刷新：本餐几项 / 累积 kcal / 宏量拆分 / 个性化目标。
  - LLM 未配置 token 时 agent 自动走规则兜底，页面仍可用（--no-llm 同理）。

排版（v3 布局重构，纯 UI 层改动，管线零接触）：
  - 主题 gr.themes.Soft(primary_hue=emerald) + 少量自定义 CSS：
    hero 区徽章、右列状态卡片化（圆角/阴影/浅底）、大数字 kcal。
  - 右列状态面板由 Markdown 文本改为 gr.HTML 卡片：热量大数字 +
    宏量进度条 + 本餐食物清单，数据仍来自 agent.meal_items（同源）。
  - 布局结构（v3 调整：输入区从"两行错位堆叠"改为统一输入区）：
      hero（紧凑标题+徽章）
      左列 = Chatbot + 输入区[ 图片 | 文字 + 发送/新一餐 ]
      右列 = 状态卡片
    图片放在输入区左侧（工作流是图片优先：先放图、再补文字说明），
    发送与新一餐同置按钮行（均为"动作"，primary/secondary 分主次），
    不再让"图片上传 × 清空会话"这两个不相关的操作绑在一行。

用法（单条命令，项目根目录运行）：
  python demo/web_demo.py            # 打开 http://127.0.0.1:7860
  python demo/web_demo.py --no-llm   # 禁用 LLM，纯规则
  python demo/web_demo.py --share    # 生成公网链接（答辩远程演示用）

Gradio 6 与 4.x 的 API 差异（本文件踩过的坑，写在这里防止回退）：
  - Chatbot 没有 type= 参数了，messages 格式（{role, content}）是唯一格式。
  - 消息 content 为 list 时每项是 {type:'text', text:...} 或 {type:'image', path:...}。
  - show_copy_button 参数没了，改 buttons=['copy']。
  - Image(type='filepath') 返回上传后的临时文件路径（str），直接喂 agent。
  - 【重要】不要把 Chatbot 组件本身作为事件输入回传：客户端（浏览器或
    gradio_client）会把 file 消息简化成裸路径字符串，回传时过不了
    pydantic 的 FileData 校验（file 字段要求 meta._type）。所以内部
    对话状态放 gr.State（纯 JSON list），Chatbot 只作输出，每轮由
    to_display() 从 state 渲染。
'''
import os
import sys
import argparse
import html as html_mod

# 与 cli_demo.py 相同的坑：直接 `python demo/web_demo.py` 启动时
# sys.path 只含 demo/，找不到根目录的 models 包，这里补上项目根。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from models.calorie_agent import CalorieAgent


def new_agent(use_llm=True):
    return CalorieAgent(use_llm=use_llm)


# ---------------- 自定义 CSS（排版美化 v2）----------------
# 为什么用 elem_id 打点而不是全局覆盖：全局选择器会被 Gradio 内部
# class 哈希（每次版本可能变）打断，elem_id 是稳定锚点。
CUSTOM_CSS = """
/* hero 区：徽章横排、呼吸感。
   注意：不要给 #hero-side 设 display:flex——列里的 HTML 组件会因此
   shrink-to-fit 收缩到内容宽，配合 flex-wrap 把徽章挤成一列一行一个。 */
#hero-row { align-items: center; }
#hero-divider {
    height: 1px; margin: 2px 0 14px 0;
    background: var(--border-color-primary);
}
#hero-badges {
    display: flex; gap: 8px; flex-wrap: wrap;
    justify-content: flex-end; align-items: center;
}
#hero-badges .badge {
    display: inline-flex; align-items: center;
    padding: 4px 12px; border-radius: 999px;
    background: rgba(16,185,129,.10); color: #047857;
    border: 1px solid rgba(16,185,129,.25);
    font-size: 13px; font-weight: 500;
}
#hero-badges .badge.gray {
    background: rgba(100,116,139,.10); color: #475569;
    border-color: rgba(100,116,139,.25);
}

/* 统一输入区：图片左格与文字右格并排 */
#input-row { align-items: stretch; }
#img-col { min-width: 220px; }

/* 右列状态卡片 */
#state-panel { padding: 0 !important; }
.state-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
}
.state-card h4 {
    margin: 0 0 8px 0; font-size: 13px; font-weight: 600;
    color: var(--body-text-color-subdued);
    letter-spacing: .04em;
}
.kcal-big { font-size: 34px; font-weight: 700; line-height: 1.1; }
.kcal-big small { font-size: 14px; font-weight: 500; color: var(--body-text-color-subdued); }
.macro-bar { height: 8px; border-radius: 4px; background: var(--background-fill-primary); overflow: hidden; margin-top: 6px; }
.macro-bar i { display: block; height: 100%; }
.food-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px; padding: 4px 10px; margin: 0 6px 6px 0;
    font-size: 13px;
}
.food-chip b { font-weight: 600; }
.food-chip .w { color: var(--body-text-color-subdued); }
.pend-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #f59e0b; margin-right: 6px;
}
"""


def _badge(text, gray=False):
    cls = "badge gray" if gray else "badge"
    return f'<span class="{cls}">{text}</span>'


def hero_html(use_llm=True):
    """hero 区徽章：能力点一目了然（答辩时观众 3 秒理解系统能做什么）。"""
    llm_badge = _badge("🤖 LLM 文本增强（规则兜底）") if use_llm \
        else _badge("⚙️ 纯规则模式", gray=True)
    return f'<div id="hero-badges">{_badge("🍽️ 50 类中餐识别")}' \
           f'{_badge("🥘 混合餐盘拆分")}{_badge("💬 多轮对话")}' \
           f'{_badge("🔌 断网可用")}{llm_badge}</div>'


def state_html(agent):
    """右侧状态面板（HTML 卡片版）：数据全部来自 agent 对话状态，
    与 CLI 状态行同源。宏量条宽度按 2000kcal 日预算的比例归一。"""
    n = len(agent.meal_items)
    kcal = sum(f["kcal"] for f in agent.meal_items)
    tot_p = sum(f["protein_g"] for f in agent.meal_items)
    tot_f = sum(f["fat_g"] for f in agent.meal_items)
    tot_c = sum(f["carbs_g"] for f in agent.meal_items)

    # 目标徽章（未设置时给中性样式）
    goal = agent.user_goal
    goal_html = (f'<span class="food-chip"><b>🎯 {html_mod.escape(goal)}</b></span>'
                 if goal else '<span class="food-chip"><span class="w">目标未设置（说"我在减脂/增肌"）</span></span>')

    # 本餐食物清单：每样一个 chip
    if n:
        chips = "".join(
            f'<span class="food-chip"><b>{html_mod.escape(f["food_name"])}</b>'
            f'<span class="w">{f["weight_g"]:.0f}g · {f["kcal"]:.0f}kcal</span></span>'
            for f in agent.meal_items)
    else:
        chips = '<span class="w" style="color:var(--body-text-color-subdued)">还没有记录，发张照片开始吧</span>'

    # 追问中提示（门控反问未收敛时亮起，提示用户回答菜名/序号）
    pend = ('<div style="margin-top:8px;font-size:12px;color:#b45309">'
            '<span class="pend-dot"></span>等待你回答菜名或序号 1/2/3'
            '</div>') if getattr(agent, "pending_clarification", None) else ""

    # 宏量进度条：蛋白/脂肪/碳水，各按克数相对展示（归一到三者最大值）
    mx = max(tot_p, tot_f, tot_c, 1)
    def bar(v, color):
        return (f'<div class="macro-bar"><i style="width:{v/mx*100:.0f}%;'
                f'background:{color}"></i></div>')

    return f"""
    <div class="state-card">
      <h4>本餐热量</h4>
      <div class="kcal-big">{kcal:.0f} <small>kcal / {n} 项</small></div>
      {pend}
    </div>
    <div class="state-card">
      <h4>宏量拆分</h4>
      <div style="font-size:13px">蛋白 {tot_p:.0f}g</div>{bar(tot_p, "#79c3aa")}
      <div style="font-size:13px">脂肪 {tot_f:.0f}g</div>{bar(tot_f, "#27d2f0")}
      <div style="font-size:13px">碳水 {tot_c:.0f}g</div>{bar(tot_c, "#0d48a7")}
    </div>
    <div class="state-card">
      <h4>本餐记录</h4>
      <div>{chips}</div>
    </div>
    <div class="state-card">
      <h4>个性化目标</h4>
      <div>{goal_html}</div>
      <div style="font-size:12px;color:var(--body-text-color-subdued);margin-top:4px">
        对话轮数 {len(agent.history)//2}
      </div>
    </div>
    """


def to_display(log):
    """内部 log（纯 JSON 可序列化）→ gradio messages 格式。

    内部格式：{'role': 'user'|'assistant', 'text': str, 'image': path|None}
    这样 gr.State 回传的永远是简单 dict，绕开 Chatbot file 消息的
    FileData/meta 校验坑（见文件头注释）。
    """
    out = []
    for m in log:
        if m.get("image"):
            out.append({"role": m["role"], "content": [
                {"type": "text", "text": m["text"]},
                {"type": "image", "path": m["image"]},
            ]})
        else:
            out.append({"role": m["role"], "content": m["text"]})
    return out


def chat_respond(text, image, agent, chat_log):
    """一次交互：图（可选）+ 文本 → agent.chat → 更新气泡与状态面板。

    chat_log 是内部格式 list（见 to_display），只在 gr.State 里流转。
    """
    text = (text or "").strip()
    if not text and image is None:
        # 空输入：不推进对话，避免把空轮写进 history 污染上下文
        return chat_log, state_html(agent)

    img_path = None
    if image is not None:
        # Gradio 给的是临时文件路径（str，纯英文 %TEMP% 下）。
        # agent 内部统一走 imread_unicode（np.fromfile+imdecode），
        # 中文路径也能读，这里无需特殊处理。
        img_path = image if isinstance(image, str) else image.name

    # 带图时给一句默认指令（与 CLI 的 `img <path>` 行为一致）
    msg = text if text else "帮我看看这盘菜"
    try:
        r = agent.chat(msg, image_path=img_path)
        reply = r["reply"]
    except Exception as e:          # 任何模块崩了都不能让页面挂掉
        reply = f"[内部错误] {e}\n（流水线异常，请换一张图或重试）"

    chat_log = chat_log + [
        {"role": "user", "text": msg, "image": img_path},
        {"role": "assistant", "text": reply, "image": None},
    ]
    return chat_log, state_html(agent)


def reset_meal(agent, chat_log):
    """新一餐按钮：清空一餐记录与对话历史，回到干净会话。"""
    agent.reset_meal()
    agent.history.clear()
    agent.user_goal = None
    return [], state_html(agent)


def build_app(use_llm=True):
    agent = new_agent(use_llm=use_llm)

    # Gradio 6 把 theme/css 从 Blocks 构造器移到了 launch()。为了保持
    # build_app() 自包含（测试里只 build 不 launch），这里持有到 launch 用。
    build_app._theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")
    build_app._css = CUSTOM_CSS

    with gr.Blocks(title="卡路里识别智能体") as app:
        # ---- hero 区：标题 + 一句话说明 + 能力徽章 ----
        # v3：hero 压缩成单行语义（大标题 + 副说明不再各占一行 Markdown 块，
        # 副说明并入徽章上方的紧凑排版），把纵向空间留给对话区。
        with gr.Row(elem_id="hero-row"):
            with gr.Column(scale=5):
                gr.Markdown(
                    "# 🍽️ 基于 VLM 的食物卡路里识别智能体\n"
                    "<span style='color:var(--body-text-color-subdued);font-size:14px'>"
                    "上传食物照片（单食物或混合餐盘均可），或直接对话：识别 → 估分量 → 算热量。</span>"
                )
            with gr.Column(scale=2, elem_id="hero-side"):
                gr.HTML(hero_html(use_llm))
        gr.HTML("<div id='hero-divider'></div>")

        with gr.Row():
            with gr.Column(scale=5, elem_id="chat-col"):
                # 对话状态存 gr.State（内部格式），Chatbot 每轮由 to_display 渲染，
                # 避免 file 消息回传的 FileData 校验坑（见文件头注释）。
                chat_state = gr.State([])
                chat = gr.Chatbot(
                    height=520,
                    label="对话",
                    buttons=["copy"],
                )
                # ---- 统一输入区：图片在左、文字在右，动作按钮一行 ----
                # v3 结构调整：原来是 [文字+发送] 与 [图片+新一餐] 两行错位堆叠，
                # 图片沉底且"上传图片"与"清空会话"这两个不相关操作被绑在一行。
                # 现在按工作流组织：图片是主要输入（占左侧独立格），
                # 文字是补充输入，发送/新一餐都是"动作"归到按钮行。
                # v3.1 尺寸修正：height=150 时粘贴区太小；且长标签
                # "（可选：单食物/混合餐盘）"在窄列里换行后与图片框重叠。
                # 标签精简为"菜品照片"（单/混合盘提示 hero 副说明已有，
                # 不丢信息），height 提到 220 给粘贴/预览留足空间。
                with gr.Row(elem_id="input-row"):
                    with gr.Column(scale=1, min_width=220, elem_id="img-col"):
                        img = gr.Image(
                            type="filepath",
                            label="菜品照片",
                            height=220,
                        )
                    with gr.Column(scale=2, elem_id="txt-col"):
                        txt = gr.Textbox(
                            placeholder="输入文字，或直接上传图片（如：帮我看看这盘菜 / 总共多少 / 再来一份 / 我在减脂）",
                            label="文字",
                            lines=3,
                        )
                        with gr.Row():
                            btn_send = gr.Button("发送", variant="primary", scale=3)
                            btn_reset = gr.Button("🆕 新一餐（清空记录）", scale=2)

            with gr.Column(scale=2, elem_id="state-panel"):
                state = gr.HTML(
                    value=state_html(agent),
                    label="📊 当前状态",
                )

        # 事件绑定：回车与按钮都触发。
        # 链式 .then()：先渲染聊天（chat_state→to_display→chat），再清空输入框。
        for src in (btn_send.click, txt.submit):
            src(
                fn=chat_respond,
                inputs=[txt, img, gr.State(agent), chat_state],
                outputs=[chat_state, state],
            ).then(
                fn=to_display, inputs=[chat_state], outputs=[chat],
            ).then(lambda: "", None, txt).then(lambda: None, None, img)
        btn_reset.click(
            fn=reset_meal,
            inputs=[gr.State(agent), chat_state],
            outputs=[chat_state, state],
        ).then(fn=to_display, inputs=[chat_state], outputs=[chat])
    return app


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM，纯规则模式")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网链接")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app(use_llm=not args.no_llm)
    app.launch(server_port=args.port, share=args.share,
               theme=build_app._theme, css=build_app._css)
