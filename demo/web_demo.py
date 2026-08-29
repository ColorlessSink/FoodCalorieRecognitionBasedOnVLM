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
  - 状态面板每轮刷新：本餐几项 / 累积 kcal / 个性化目标。
  - LLM 未配置 token 时 agent 自动走规则兜底，页面仍可用（--no-llm 同理）。

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

# 与 cli_demo.py 相同的坑：直接 `python demo/web_demo.py` 启动时
# sys.path 只含 demo/，找不到根目录的 models 包，这里补上项目根。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from models.calorie_agent import CalorieAgent


def new_agent(use_llm=True):
    return CalorieAgent(use_llm=use_llm)


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
        return chat_log, state_text(agent)

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
    return chat_log, state_text(agent)


def state_text(agent):
    """右侧状态面板：一餐累积 + 目标（与 CLI 底部状态行同源）。"""
    n = len(agent.meal_items)
    kcal = sum(f["kcal"] for f in agent.meal_items)
    last = agent.last_food["food_name"] if agent.last_food else "—"
    goal = agent.user_goal or "未设置"
    lines = [
        f"**本餐记录**：{n} 项 / {kcal:.0f} kcal",
        f"**最近食物**：{last}",
        f"**个性化目标**：{goal}",
        f"**对话轮数**：{len(agent.history) // 2}",
    ]
    return "\n\n".join(lines)


def reset_meal(agent, chat_log):
    """新一餐按钮：清空一餐记录与对话历史，回到干净会话。"""
    agent.reset_meal()
    agent.history.clear()
    agent.user_goal = None
    return [], state_text(agent)


def build_app(use_llm=True):
    agent = new_agent(use_llm=use_llm)

    with gr.Blocks(title="卡路里识别智能体") as app:
        gr.Markdown(
            "# 🍽️ 基于 VLM 的食物卡路里识别智能体\n"
            "上传食物照片（单食物或混合餐盘均可），或直接对话：识别 → 估分量 → 算热量。\n"
            "支持多轮追问（\"再来一份\"\"总共多少\"\"我在减脂\"），规则兜底，断网可用。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                # 对话状态存 gr.State（内部格式），Chatbot 每轮由 to_display 渲染，
                # 避免 file 消息回传的 FileData 校验坑（见文件头注释）。
                chat_state = gr.State([])
                chat = gr.Chatbot(
                    height=480,
                    label="对话",
                    buttons=["copy"],
                )
                with gr.Row():
                    txt = gr.Textbox(
                        placeholder="输入文字，或直接上传图片（如：帮我看看这盘菜 / 总共多少 / 再来一份 / 我在减脂）",
                        label="文字",
                        scale=4,
                    )
                    btn_send = gr.Button("发送", variant="primary", scale=1)
                with gr.Row():
                    img = gr.Image(
                        type="filepath",
                        label="菜品照片（可选：单食物 / 混合餐盘）",
                        height=180,
                    )
                    with gr.Column():
                        btn_reset = gr.Button("🆕 新一餐（清空记录）")
                        llm_note = gr.Markdown(
                            ("LLM：已启用（glm-5.2，纯文本生成，规则兜底）" if use_llm
                             else "LLM：已禁用（纯规则模式）")
                        )
            with gr.Column(scale=1):
                gr.Markdown("### 📊 当前状态")
                state = gr.Markdown(state_text(agent))

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
    app.launch(server_port=args.port, share=args.share)

