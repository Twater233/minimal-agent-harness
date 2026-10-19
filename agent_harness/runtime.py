import json
import time
import uuid

from .context import ContextLimit, ContextManager
from .llm import LLMError
from .tools import ToolContext, build_registry


def close_pending(history):
    pending = {}
    for message in history:
        if message["role"] == "assistant":
            for call in message.get("tool_calls", []):
                pending[call["id"]] = call
        elif message["role"] == "tool":
            pending.pop(message["tool_call_id"], None)
    for call_id in pending:
        history.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps({
            "ok": False, "error": {"type": "Interrupted", "message":
                "Execution was interrupted; side effect may have occurred. Inspect current state before repeating any mutation."}})})
    return len(pending)


class Agent:
    def __init__(self, store, llm, registry=None, max_steps=8, context_budget=60000):
        if not 1 <= max_steps <= 20:
            raise ValueError("max_steps must be between 1 and 20")
        self.store = store
        self.llm = llm
        self.registry = registry or build_registry()
        self.max_steps = max_steps
        self.context = ContextManager(llm, context_budget)

    def run(self, user, session, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Input must contain 1..4000 characters")
        if not user or not session or len(user) > 100 or len(session) > 100:
            raise ValueError("User and session must contain 1..100 characters")
        turn = uuid.uuid4().hex
        self.store.acquire(user, session, turn)
        summary, history = self.store.load(user, session)
        ctx = ToolContext(self.store, user, session)

        def checkpoint():
            self.store.save(user, session, turn, summary, history)

        def log(kind, data):
            self.store.event(user, session, turn, kind, data)

        try:
            recovered = close_pending(history)
            if recovered:
                log("recovery", {"unresolved_calls": recovered})
            history.append({"role": "user", "content": text.strip()})
            checkpoint()
            log("turn_start", {"max_steps": self.max_steps})
            schemas = self.registry.schemas()
            for step in range(self.max_steps):
                self.store.renew(user, session, turn)
                memory = self.store.todo(user, session, "list")
                old_length = len(history)
                summary, history, messages, _ = self.context.fit(summary, history, memory, schemas)
                if len(history) < old_length:
                    log("context_compacted", {"removed_messages": old_length - len(history)})
                    checkpoint()
                started = time.monotonic()
                decision = self.llm.complete(messages, schemas)
                self.store.renew(user, session, turn)
                log("model_decision", {"step": step + 1, "kind": decision.kind,
                                       "duration_ms": round((time.monotonic() - started) * 1000),
                                       "action_note": decision.action_note[:300]})
                history.append(decision.message)
                checkpoint()
                if decision.kind == "final":
                    log("turn_end", {"status": "completed"})
                    return decision.message["content"]
                # Preserve model order; todo writes can depend on an earlier call.
                for call in decision.message["tool_calls"]:
                    self.store.renew(user, session, turn)
                    name = call["function"]["name"]
                    log("tool_start", {"call_id": call["id"], "name": name})
                    started = time.monotonic()
                    result = self.registry.execute(name, call["function"]["arguments"], ctx)
                    encoded = json.dumps(result, ensure_ascii=False)
                    if len(encoded) > 18000:
                        encoded = json.dumps({"ok": False, "error": {"type": "OutputLimit",
                                             "message": "Tool executed but result exceeds 18000 characters; narrow the query before continuing."}})
                    history.append({"role": "tool", "tool_call_id": call["id"], "content": encoded})
                    checkpoint()
                    log("tool_end", {"call_id": call["id"], "name": name, "ok": result["ok"],
                                     "output_limited": len(json.dumps(result, ensure_ascii=False)) > 18000,
                                     "result": json.loads(encoded),
                                     "duration_ms": round((time.monotonic() - started) * 1000)})
            answer = "已达到本轮工具循环上限，已完成的操作已保存。请缩小任务或继续提问；尚未完成的部分不能视为成功。"
            history.append({"role": "assistant", "content": answer})
            checkpoint()
            log("turn_end", {"status": "step_limit"})
            return answer
        except (LLMError, ContextLimit) as exc:
            close_pending(history)
            answer = "本轮未完成：" + str(exc) + "。已执行的操作会保留，请先查看待办再重试。"
            history.append({"role": "assistant", "content": answer})
            checkpoint()
            log("turn_end", {"status": "error", "error_type": type(exc).__name__})
            return answer
        finally:
            self.store.release(user, session, turn)
