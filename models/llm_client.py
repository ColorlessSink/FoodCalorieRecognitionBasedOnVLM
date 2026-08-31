'''
glm-5.2 LLM 调用客户端模块（OpenAI 兼容）
---
只做纯文本任务（实测代理视觉不可用）：CoT 分量推理、多轮对话

要点：
  - 代理 glm-5.2 是"带 thinking 的模型"，输出前先内部推理（reasoning_tokens 很大），
    若 max_tokens 给小了，输出会被推理过程吃光，content 返回 None/空
    经验值：max_tokens ≥ 512 才稳定能拿到正文
  - 所有调用都必须有规则兜底（超时/None content → 返回 fallback），
    保证下游流水线不因网络抖动中断
'''
import os, sys, json, re
import requests

# `python models/llm_client.py` 启动时 sys.path 只有 models/，补上项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import load_config


class LLMClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        llm = self.cfg["llm"]
        # base_url/model may be environment-variable names or literal values.
        self.base_url = self._resolve_env_or_value(llm.get("base_url", "")).rstrip("/")
        self.model = self._resolve_env_or_value(llm.get("model", ""))
        token_key = llm.get("token_env", "ANTHROPIC_AUTH_TOKEN")
        self.token = os.environ.get(token_key, "")
        self.timeout = llm.get("timeout", 30)
        self.max_tokens = llm.get("max_tokens", 1024)
        self.temperature = llm.get("temperature", 0.3)
        self.fallback = llm.get("fallback_to_rules", True)
        self.enabled = (llm.get("enabled", True)
                        and bool(self.base_url)
                        and bool(self.model)
                        and bool(self.token))
        # Claude/DeepSeek 配置通常把 base URL 指向 /anthropic，
        # 此时应使用 Anthropic Messages 协议，而不是 OpenAI Chat Completions。
        self.anthropic_compat = self.base_url.rstrip("/").endswith("/anthropic")

    @staticmethod
    def _resolve_env_or_value(value):
        """Resolve an environment-variable name, or keep a literal value."""
        if value is None:
            return ""
        value = str(value).strip()
        if value in os.environ:
            return os.environ[value].strip()
        # 配置中的大写标识符通常表示“环境变量名”。变量缺失时返回空值，
        # 避免把 ANTHROPIC_BASE_URL 之类的名字误当成实际 URL。
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
            return ""
        return value

    def chat(self, messages, max_tokens=None, temperature=None, fallback=""):
        # messages: OpenAI 格式 [{"role":"user","content":"..."},...]
        # 成功返回 content 字符串；失败且 self.fallback 则返回 fallback
        if not self.enabled:
            return fallback
        if self.anthropic_compat:
            url = self.base_url + "/v1/messages"
            hdr = {
                "x-api-key": self.token,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            system = "\n".join(
                str(m.get("content", "")) for m in messages
                if m.get("role") == "system"
            ).strip()
            body_messages = [m for m in messages if m.get("role") != "system"]
            body = {
                "model": self.model,
                "messages": body_messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": self.temperature if temperature is None else temperature,
            }
            if system:
                body["system"] = system
        else:
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
            if self.anthropic_compat:
                blocks = data.get("content") or []
                content = "\n".join(
                    str(block.get("text", "")) for block in blocks
                    if block.get("type") == "text"
                ).strip()
            else:
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
        # 单轮便捷调用
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    c = LLMClient()
    print("enabled:", c.enabled)
    if c.enabled:
        ans = c.ask("只回复四个字：测试通过，不要思考过程", max_tokens=512, fallback="[LLM失败]")
        print("回复:", ans)
