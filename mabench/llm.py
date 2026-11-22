"""LLM clients. `VertexLLM` is the only code that can spend money."""
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import ALLOWED_PROJECT, CREDIT_CONFIRMED_ENV, GCLOUD_CONFIG

RETRYABLE = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
TOKEN_TTL_SECONDS = 1800


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: list | None = None


def call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name, args)])


class FakeLLM:
    """Replays scripted responses; tokens are estimated so budgets behave realistically."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, tools, max_output_tokens, temperature):
        if self.calls >= len(self.script):
            raise RuntimeError("FakeLLM script exhausted")
        r = self.script[self.calls]
        self.calls += 1
        out = len(r.text) + len(json.dumps([c.args for c in r.tool_calls]))
        return LLMResponse(r.text, r.tool_calls, sum(len(m.get("content") or "") for m in messages) // 4, out // 4 + 1)


def to_gemini(messages):
    """Convert internal messages; assistant parts replay verbatim to keep Gemini thought signatures."""
    system, contents = None, []
    for m in messages:
        role = m["role"]
        if role == "system":
            system = {"parts": [{"text": m["content"]}]}
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif role == "assistant":
            parts = m.get("raw")
            if parts is None:
                parts = [{"text": m["content"]}] if m.get("content") else []
                parts += [{"functionCall": {"name": c.name, "args": c.args}} for c in m.get("tool_calls", [])]
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            part = {"functionResponse": {"name": m["name"], "response": {"output": m["content"]}}}
            if contents and contents[-1]["role"] == "user" and "functionResponse" in contents[-1]["parts"][0]:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
    return system, contents


def parse_gemini(data: dict) -> LLMResponse:
    candidates = data.get("candidates") or [{}]
    parts = (candidates[0].get("content") or {}).get("parts") or []
    usage = data.get("usageMetadata") or {}
    prompt = usage.get("promptTokenCount", 0)
    output = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    text = "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    calls = [ToolCall(p["functionCall"]["name"], p["functionCall"].get("args") or {})
             for p in parts if "functionCall" in p]
    return LLMResponse(text, calls, prompt, output, parts)


class VertexLLM:
    def __init__(self, model: str, project: str = ALLOWED_PROJECT, location: str = "global"):
        if project != ALLOWED_PROJECT:
            raise PermissionError(f"refusing project {project!r}; only {ALLOWED_PROJECT} is allowed")
        if os.environ.get(CREDIT_CONFIRMED_ENV) != "1":
            raise PermissionError(f"paid calls are locked; set {CREDIT_CONFIRMED_ENV}=1 once the credit is confirmed")
        self.url = (f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
                    f"/publishers/google/models/{model}:generateContent")
        self.project = project
        self._token, self._token_at = None, 0.0

    def _access_token(self, refresh=False) -> str:
        if refresh or not self._token or time.time() - self._token_at > TOKEN_TTL_SECONDS:
            gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
            if not gcloud:
                raise RuntimeError("gcloud not found")
            env = {**os.environ, "CLOUDSDK_ACTIVE_CONFIG_NAME": GCLOUD_CONFIG}
            self._token = subprocess.run([gcloud, "auth", "print-access-token"], env=env, check=True,
                                         capture_output=True, text=True, timeout=60).stdout.strip()
            self._token_at = time.time()
        return self._token

    def complete(self, messages, tools, max_output_tokens, temperature) -> LLMResponse:
        system, contents = to_gemini(messages)
        body = {"contents": contents, "tools": [{"functionDeclarations": tools}],
                "generationConfig": {"maxOutputTokens": max_output_tokens, "temperature": temperature}}
        if system:
            body["systemInstruction"] = system
        payload = json.dumps(body).encode()
        refreshed = False
        for attempt in range(MAX_ATTEMPTS):
            req = urllib.request.Request(self.url, payload, {
                "Authorization": f"Bearer {self._access_token()}", "Content-Type": "application/json",
                "x-goog-user-project": self.project})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return parse_gemini(json.load(r))
            except urllib.error.HTTPError as e:
                if e.code == 401 and not refreshed:
                    self._access_token(refresh=True)
                    refreshed = True
                    continue
                if e.code not in RETRYABLE or attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError(f"Vertex HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError(f"Vertex connection failed: {e}") from e
            time.sleep(2 ** attempt)
        raise RuntimeError("Vertex retries exhausted")
