'''
models/llm_client.py — glm-5.2 LLM 调用客户端（OpenAI 兼容）
================================================
为什么需要单独封装：
  - 代理 glm-5.2 是"带 thinking 的模型"，输出前先内部推理（reasoning_tokens 很大），
    若 max_tokens 给小了，输出会被推理过程吃光，content 返回 None/空。
    经验值：max_tokens ≥ 512 才稳定能拿到正文（见 process_log.md §0.3）。
  - 所有调用都必须有规则兜底（超时/None content → 返回 fallback），
    保证下游流水线不因网络抖动中断。
  - 中文在 Windows 控制台是 gbk 乱码，但程序内部用 utf-8 字符串没问题，
    只在 print 时需注意（这里不 print，交给上层）。

只做纯文本任务（实测代理视觉不可用）：CoT 分量推理、多轮对话。
'''
import os
import json
import requests

from common import load_config


class LLMClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        llm = self.cfg["llm"]
        self.base_url = llm["base_url"].rstrip("/")
        self.model = llm["model"]
        self.token = os.environ.get(llm["token_env"], "")
        self.timeout = llm.get("timeout", 30)
        self.max_tokens = llm.get("max_tokens", 1024)
        self.temperature = llm.get("temperature", 0.3)
        self.fallback = llm.get("fallback_to_rules", True)
        self.enabled = llm.get("enabled", True) and bool(self.token)

    def chat(self, messages, max_tokens=None, temperature=None, fallback=""):
        """
        messages: OpenAI 格式 [{"role":"user","content":"..."},...]
        成功返回 content 字符串；失败且 self.fallback 则返回 fallback。
        """
        if not self.enabled:
            return fallback
        url = self.base_url + "/v1/chat/completions"
        hdr = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        try:
            r = requests.post(url, headers=hdr, json=body, timeout=self.timeout)
            if r.status_code != 200:
                return fallback
            data = r.json()
            content = data["choices"][0]["message"].get("content")
            if not content:
                # 推理把 token 吃光的情况：加大 max_tokens 重试一次
                if (max_tokens or self.max_tokens) < 1024:
                    return self.chat(messages, max_tokens=1024, temperature=temperature, fallback=fallback)
                return fallback
            return content.strip()
        except Exception:
            return fallback

    def ask(self, prompt, system=None, **kw):
        """单轮便捷调用。"""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    c = LLMClient()
    print("enabled:", c.enabled)
    if c.enabled:
        ans = c.ask("只回复四个字：测试通过，不要思考过程", max_tokens=512, fallback="[LLM失败]")
        print("回复:", ans)
