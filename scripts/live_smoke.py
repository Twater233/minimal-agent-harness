"""Opt-in real API acceptance test. Requires an installed package and .env."""
import datetime
import json
import os
import tempfile
import uuid
from pathlib import Path

from agent_harness.llm import OpenAICompatible, load_env
from agent_harness.runtime import Agent
from agent_harness.store import Store


def main():
    load_env()
    report = {"model": os.getenv("LLM_MODEL", ""), "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "failed", "checks": [], "answers": []}
    try:
        llm = OpenAICompatible.from_env()
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "live.db")
            agent = Agent(store, llm)
            marker = "LIVE-" + uuid.uuid4().hex[:8]
            report["answers"].append(agent.run("A", "window-1",
                f"请用 weather 查询厦门演示天气，并用 todo 添加待办：{marker}。最后明确说明天气是否是真实预报。"))
            items = store.todo("A", "window-1", "list")["items"]
            assert len(items) == 1 and marker in items[0]["title"], "todo add missing or duplicated"
            report["checks"].append("todo_add")
            events = store.events("A", "window-1")
            assert any(e["kind"] == "tool_end" and e["data"]["name"] == "weather" and
                       e["data"]["ok"] for e in events), "weather tool was not executed successfully"
            assert any(word in report["answers"][-1] for word in ("模拟", "演示", "mock", "非真实", "不是真实")), "mock not disclosed"
            report["checks"].append("weather_and_mock_disclosure")
            report["answers"].append(agent.run("A", "window-1", "把刚才的待办标记完成，然后用计算器算 (125+75)*3。"))
            assert store.todo("A", "window-1", "list")["items"][0]["done"] == 1, "followup todo incomplete"
            assert any(e["kind"] == "tool_end" and e["data"]["name"] == "calculator" and
                       e["data"]["result"].get("data", {}).get("value") == 600
                       for e in store.events("A", "window-1")), "calculator result missing"
            report["checks"].append("tool_followup_and_calculator")
            report["answers"].append(agent.run("A", "window-2", "请调用 todo list，告诉我当前窗口有哪些待办。"))
            assert not store.todo("A", "window-2", "list")["items"], "session leak"
            assert any(e["kind"] == "tool_end" and e["data"]["name"] == "todo" and e["data"]["ok"]
                       for e in store.events("A", "window-2")), "second session tool missing"
            assert marker not in report["answers"][-1], "first-session content leaked into second-session reply"
            assert not store.todo("B", "window-1", "list")["items"], "user leak"
            report["checks"].append("session_and_user_isolation")
            report["answers"].append(agent.run("A", "window-1", "刚才的计算结果再加十是多少？"))
            assert "610" in report["answers"][-1], "plain followup lost context"
            report["checks"].append("plain_followup")
            report["status"] = "passed"
    except Exception as exc:
        # Avoid logging provider bodies, credentials, or arbitrary exception payloads.
        report["error_type"] = type(exc).__name__
        raise
    finally:
        output = Path(".agent/live-smoke.json")
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Real API acceptance: {report['status']}; report: {output.resolve()}")


if __name__ == "__main__":
    main()
