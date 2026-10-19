import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class LLMError(RuntimeError):
    pass


class ProtocolError(LLMError):
    pass


@dataclass
class Decision:
    message: dict
    kind: str
    action_note: str


def parse_response(payload):
    try:
        choice = payload["choices"][0]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise ProtocolError("Model response truncated or filtered")
        source = choice["message"]
        content = source.get("content") or ""
        if not isinstance(content, str) or len(content) > 12000:
            raise ProtocolError("Invalid or oversized message content")
        calls = source.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) > 4:
            raise ProtocolError("At most four tool calls per model response")
        message = {"role": "assistant", "content": content or None}
        ids = set()
        normalized = []
        for call in calls:
            call_id = call["id"]
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            if (call.get("type") != "function" or not isinstance(call_id, str)
                    or not call_id or len(call_id) > 128 or call_id in ids
                    or not isinstance(name, str) or not name or len(name) > 100
                    or not isinstance(arguments, str) or len(arguments) > 8000):
                raise ProtocolError("Malformed tool call")
            ids.add(call_id)
            normalized.append({"id": call_id, "type": "function",
                               "function": {"name": name, "arguments": arguments}})
        if normalized:
            message["tool_calls"] = normalized
        elif not content.strip():
            raise ProtocolError("Empty model response")
        # Only keep the visible action note, never reasoning_content or hidden chain of thought.
        return Decision(message, "tools" if normalized else "final", content if normalized else "")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProtocolError("Malformed chat completion response") from exc


def load_env(path=".env"):
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class OpenAICompatible:
    def __init__(self, base_url, api_key, model, timeout=40):
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("LLM_BASE_URL must be an HTTPS URL without credentials, query, or fragment")
        if not api_key or not model:
            raise ValueError("Set LLM_API_KEY and LLM_MODEL in .env or environment")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls):
        load_env()
        return cls(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                   os.getenv("LLM_API_KEY", ""), os.getenv("LLM_MODEL", ""))

    def complete(self, messages, tools):
        body = {"model": self.model, "messages": messages, "max_completion_tokens": 2048}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + self.api_key,
                                                  "Content-Type": "application/json"})
        # Never forward an API key through a redirect to another host.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(NoRedirect())
        for attempt in range(2):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    data = response.read(1_000_001)
                if len(data) > 1_000_000:
                    raise ProtocolError("API response exceeds 1 MB")
                return parse_response(json.loads(data))
            except urllib.error.HTTPError as exc:
                if attempt == 0 and exc.code in (429, 500, 502, 503, 504):
                    time.sleep(0.5)
                    continue
                # Do not expose provider response bodies or request headers.
                raise LLMError(f"LLM HTTP {exc.code}; verify provider, model, quota, and credentials") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                raise LLMError("LLM connection failed or timed out") from None
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ProtocolError("LLM response is not valid JSON") from None
