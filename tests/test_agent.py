import copy
import json
import os
import tempfile
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import Mock, patch

from agent_harness.context import ContextManager, SYSTEM, size
from agent_harness.llm import LLMError, OpenAICompatible, ProtocolError, load_env, parse_response
from agent_harness.runtime import Agent
from agent_harness.store import BusyError, Store
from agent_harness.tools import ToolContext, build_registry, calculate


def final(text):
    return parse_response({"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


def calls(*items):
    return parse_response({"choices": [{"message": {"content": "查询并记录。", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": name,
         "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}
        for i, (name, arguments) in enumerate(items)]}, "finish_reason": "tool_calls"}]})


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if not self.responses:
            raise AssertionError("Unexpected model request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.name, "state.db")
        self.store = Store(self.path)

    def agent(self, *responses, **kwargs):
        llm = ScriptedLLM(*responses)
        return Agent(self.store, llm, **kwargs), llm

    def test_direct_reply_and_plain_followup(self):
        agent, llm = self.agent(final("你好，小林。"), final("你叫小林。"))
        agent.run("A", "one", "我叫小林")
        self.assertEqual(agent.run("A", "one", "我叫什么？"), "你叫小林。")
        self.assertIn({"role": "user", "content": "我叫小林"}, llm.requests[1][0])
        self.assertEqual({t["function"]["name"] for t in llm.requests[0][1]},
                         {"calculator", "search", "todo", "weather"})

    def test_multi_tool_loop_then_tool_followup(self):
        agent, llm = self.agent(
            calls(("weather", {"city": "厦门"}), ("todo", {"action": "add", "title": "带伞"})),
            final("模拟天气晴，已记录带伞。"),
            calls(("todo", {"action": "complete", "id": 1})), final("已完成。"))
        agent.run("A", "one", "查厦门天气，记个带伞待办")
        results = [m for m in llm.requests[1][0] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in results], ["call_0", "call_1"])
        self.assertTrue(json.loads(results[0]["content"])["data"]["mock"])
        agent.run("A", "one", "把刚才的待办标记完成")
        self.assertEqual(self.store.todo("A", "one", "list")["items"][0]["done"], 1)
        self.assertTrue(any("带伞" in (m.get("content") or "") for m in llm.requests[2][0]))

    def test_user_and_session_isolation_with_restart(self):
        agent, _ = self.agent(calls(("todo", {"action": "add", "title": "写周报"})), final("已添加"))
        agent.run("A", "one", "记待办")
        self.assertEqual(self.store.todo("A", "two", "list")["items"], [])
        self.assertEqual(self.store.todo("B", "one", "list")["items"], [])
        with self.assertRaises(ValueError):
            self.store.todo("B", "one", "delete", item_id=1)
        llm = ScriptedLLM(final("你有写周报的待办"))
        Agent(Store(self.path), llm).run("A", "one", "继续")
        self.assertTrue(any("写周报" in (m.get("content") or "") for m in llm.requests[0][0]))
        isolated, isolated_llm = self.agent(final("你好"))
        isolated.run("A", "two", "你好")
        self.assertNotIn("写周报", json.dumps(isolated_llm.requests, ensure_ascii=False))

    def test_bad_arguments_unknown_tool_and_recovery(self):
        agent, llm = self.agent(
            calls(("todo", "{bad json")),
            calls(("missing", {})),
            calls(("calculator", {"expression": "6*7"})), final("42"))
        self.assertEqual(agent.run("A", "one", "算一下"), "42")
        results = [json.loads(m["content"]) for m in llm.requests[-1][0] if m["role"] == "tool"]
        self.assertEqual([r["ok"] for r in results], [False, False, True])
        self.assertEqual(results[-1]["data"]["value"], 42)

    def test_iteration_limit_closes_protocol(self):
        agent, _ = self.agent(calls(("calculator", {"expression": "1+1"})), max_steps=1)
        self.assertIn("上限", agent.run("A", "one", "持续调用"))
        _, history = self.store.load("A", "one")
        self.assertEqual([m["role"] for m in history], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(self.store.events("A", "one")[-1]["data"]["status"], "step_limit")

    def test_provider_failure_preserves_history_and_releases_lock(self):
        agent, _ = self.agent(LLMError("LLM HTTP 401"))
        self.assertIn("401", agent.run("A", "one", "你好"))
        self.assertEqual(self.store.load("A", "one")[1][0]["content"], "你好")
        self.store.acquire("A", "one", "new")

    def test_busy_rejects_same_session_but_not_other_session(self):
        self.store.acquire("A", "one", "owner")
        agent, llm = self.agent(final("ok"))
        with self.assertRaises(BusyError):
            agent.run("A", "one", "new")
        self.assertEqual(llm.requests, [])
        self.assertEqual(agent.run("A", "two", "new"), "ok")
        self.store.release("A", "one", "wrong-owner")
        with self.assertRaises(BusyError):
            self.store.acquire("A", "one", "another")

    def test_expired_lease_is_fenced(self):
        self.store.acquire("A", "one", "old")
        with self.store.connect() as db:
            db.execute("UPDATE sessions SET lease_until=0")
        self.store.acquire("A", "one", "new")
        with self.assertRaises(BusyError):
            self.store.save("A", "one", "old", "", [])
        with self.assertRaises(BusyError):
            self.store.renew("A", "one", "old")

    def test_interrupted_tool_is_not_reexecuted(self):
        self.store.acquire("A", "one", "old")
        self.store.todo("A", "one", "add", "只加一次")
        pending = calls(("todo", {"action": "add", "title": "只加一次"})).message
        self.store.save("A", "one", "old", "", [{"role": "user", "content": "记待办"}, pending])
        self.store.release("A", "one", "old")
        agent, llm = self.agent(final("已检查"))
        agent.run("A", "one", "继续")
        self.assertEqual(len(self.store.todo("A", "one", "list")["items"]), 1)
        results = [m for m in llm.requests[0][0] if m["role"] == "tool"]
        self.assertEqual(json.loads(results[0]["content"])["error"]["type"], "Interrupted")

    def test_context_compaction_keeps_complete_blocks_and_memory(self):
        llm = ScriptedLLM(final("用户叫小林，待完成周报。"))
        manager = ContextManager(llm, budget=6500)
        history = []
        for i in range(12):
            history += [{"role": "user", "content": f"问题{i}" + "旧" * 300},
                        {"role": "assistant", "content": "答" * 300}]
        history += [{"role": "user", "content": "现在继续"},
                    calls(("calculator", {"expression": "1+1"})).message,
                    {"role": "tool", "tool_call_id": "call_0", "content": "2"}]
        memory = {"items": [{"id": 1, "title": "写周报", "done": 0}]}
        summary, recent, messages, compacted = manager.fit("旧摘要", history, memory, [])
        self.assertTrue(compacted)
        self.assertIn("小林", summary)
        self.assertEqual(recent[-3:], history[-3:])
        self.assertLessEqual(size({"messages": messages, "tools": []}), 6500)
        self.assertIn("写周报", json.dumps(messages, ensure_ascii=False))

    def test_summary_failure_has_honest_fallback(self):
        manager = ContextManager(ScriptedLLM(LLMError("timeout")))
        summary = manager.summarize("早期目标", [{"role": "user", "content": "继续写周报"}])
        self.assertIn("摘要服务失败", summary)
        self.assertIn("写周报", summary)

    def test_context_overflow_never_calls_model(self):
        agent, llm = self.agent(context_budget=100)
        self.assertIn("context budget", agent.run("A", "one", "你好"))
        self.assertEqual(llm.requests, [])

    def test_todo_memory_recalled_each_turn(self):
        agent, llm = self.agent(final("hello"))
        self.store.todo("A", "one", "add", "持久化任务")
        agent.run("A", "one", "我还有什么事？")
        self.assertIn("持久化任务", json.dumps(llm.requests[0], ensure_ascii=False))
        self.assertEqual(llm.requests[0][0][0], {"role": "system", "content": SYSTEM})

    def test_trace_has_call_ids_results_and_status(self):
        agent, _ = self.agent(calls(("calculator", {"expression": "1/0"})), final("除零无效"))
        agent.run("A", "one", "除零")
        events = self.store.events("A", "one")
        end = next(e for e in events if e["kind"] == "tool_end")
        self.assertFalse(end["data"]["ok"])
        self.assertEqual(end["data"]["call_id"], "call_0")
        self.assertIn("duration_ms", end["data"])
        self.assertEqual(len({e["turn_id"] for e in events}), 1)

    def test_oversized_tool_output_does_not_get_forwarded(self):
        registry = build_registry()
        registry.register("large", "Large test result", {"type": "object"},
                          lambda context: {"blob": "x" * 20000})
        agent, llm = self.agent(calls(("large", {})), final("请缩小查询"), registry=registry)
        agent.run("A", "one", "large result")
        result = next(m for m in llm.requests[1][0] if m["role"] == "tool")
        self.assertEqual(json.loads(result["content"])["error"]["type"], "OutputLimit")
        self.assertLess(len(result["content"]), 1000)
        self.assertTrue(next(e for e in self.store.events("A", "one") if e["kind"] == "tool_end")["data"]["output_limited"])

    def test_todo_delete_missing_title_and_cross_session_id(self):
        context = ToolContext(self.store, "A", "one")
        registry = build_registry()
        self.assertFalse(registry.execute("todo", '{"action":"add"}', context)["ok"])
        added = registry.execute("todo", '{"action":"add","title":"test"}', context)
        item_id = added["data"]["affected_id"]
        other = ToolContext(self.store, "A", "two")
        self.assertFalse(registry.execute("todo", json.dumps({"action": "delete", "id": item_id}), other)["ok"])
        self.assertTrue(registry.execute("todo", json.dumps({"action": "delete", "id": item_id}), context)["ok"])
        self.assertEqual(self.store.todo("A", "one", "list")["items"], [])

    def test_blank_and_oversized_input_rejected(self):
        agent, _ = self.agent()
        for value in (" ", "x" * 4001):
            with self.assertRaises(ValueError):
                agent.run("A", "one", value)

    def test_two_hundred_turns_remain_bounded_and_recall_todos(self):
        class RepeatingLLM:
            def __init__(self):
                self.summary_calls = 0
                self.chat_calls = 0

            def complete(inner, messages, tools):
                if not tools:
                    inner.summary_calls += 1
                    return final("用户在讨论项目方案；保留当前待办，以最新状态为准。")
                inner.chat_calls += 1
                self.assertLessEqual(size({"messages": messages, "tools": tools}), 10000)
                self.assertIn("长期任务", json.dumps(messages, ensure_ascii=False))
                self.assertEqual(messages[-1]["role"], "user")
                return final("已继续讨论。" * 30)

        llm = RepeatingLLM()
        self.store.todo("A", "long", "add", "长期任务")
        agent = Agent(self.store, llm, context_budget=10000)
        for i in range(200):
            agent.run("A", "long", f"第{i}轮：" + "请继续讨论项目。" * 30)
        self.assertEqual(llm.chat_calls, 200)
        self.assertGreater(llm.summary_calls, 0)
        summary, history = self.store.load("A", "long")
        self.assertTrue(summary)
        self.assertLess(len(history), 20)
        self.assertIn("第199轮", history[-2]["content"])


class ToolAndProtocolTests(unittest.TestCase):
    def test_calculator_accuracy_and_restricted_syntax(self):
        self.assertEqual(calculate("-(3+2)*4 + 2**3")["value"], -12)
        for expression in ("__import__('os')", "(1).__class__", "True+1", "2**9999", "1/0", "1e309", "(-1)**0.5"):
            with self.subTest(expression=expression):
                with self.assertRaises((ValueError, ArithmeticError)):
                    calculate(expression)

    def test_schema_rejects_wrong_type_extra_fields_and_invalid_actions(self):
        registry = build_registry()
        for name, arguments in (("calculator", {"expression": 12}),
                                ("calculator", {"expression": "1", "extra": True}),
                                ("todo", {"action": "drop"}),
                                ("todo", {"action": "complete", "id": True}),
                                ("todo", [])):
            with self.subTest(arguments=arguments):
                self.assertFalse(registry.execute(name, json.dumps(arguments), None)["ok"])

    def test_duplicate_registration_rejected(self):
        registry = build_registry()
        with self.assertRaises(ValueError):
            registry.register("todo", "duplicate", {"type": "object"}, None)

    def test_mock_search_no_fabricated_fallback_and_weather_disclosure(self):
        registry = build_registry()
        found = registry.execute("search", json.dumps({"query": "周报"}), None)
        self.assertTrue(found["data"]["mock"])
        self.assertTrue(found["data"]["results"])
        empty = registry.execute("search", json.dumps({"query": "no-such-topic"}), None)
        self.assertEqual(empty["data"]["results"], [])
        weather = registry.execute("weather", json.dumps({"city": "厦门"}), None)
        self.assertTrue(weather["data"]["mock"])

    def test_parser_rejects_truncated_empty_duplicate_and_malformed_responses(self):
        invalid = [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]},
                   {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
                   {"choices": [{"message": {"tool_calls": "bad"}}]}]
        duplicated = calls(("todo", {"action": "list"}), ("todo", {"action": "list"})).message
        duplicated["tool_calls"][1]["id"] = duplicated["tool_calls"][0]["id"]
        invalid.append({"choices": [{"message": duplicated}]})
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ProtocolError):
                    parse_response(payload)

    def test_reasoning_content_is_not_stored(self):
        decision = parse_response({"choices": [{"message": {"content": "答案",
                                                                     "reasoning_content": "private"}}]})
        self.assertNotIn("private", json.dumps(decision.message))

    def test_config_requires_https_model_and_key(self):
        for base, key, model in (("http://example.com/v1", "x", "m"),
                                 ("https://example.com/v1", "", "m"),
                                 ("https://example.com/v1", "x", "")):
            with self.assertRaises(ValueError):
                OpenAICompatible(base, key, model)

    def test_env_does_not_override_existing_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, ".env")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("HARNESS_TEST_VALUE=file\n")
            with patch.dict(os.environ, {"HARNESS_TEST_VALUE": "environment"}):
                load_env(path)
                self.assertEqual(os.environ["HARNESS_TEST_VALUE"], "environment")


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.client = OpenAICompatible("https://example.com/v1", "secret-test-key", "test-model")
        self.opener = Mock()
        patcher = patch("urllib.request.build_opener", return_value=self.opener)
        self.build = patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def response(data):
        response = Mock()
        response.__enter__ = Mock(return_value=BytesIO(data))
        response.__exit__ = Mock(return_value=False)
        return response

    def test_real_wire_shape_and_auth_header(self):
        self.opener.open.return_value = self.response(b'{"choices":[{"message":{"content":"ok"}}]}')
        schemas = build_registry().schemas()
        decision = self.client.complete([{"role": "user", "content": "hello"}], schemas)
        self.assertEqual(decision.kind, "final")
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["tools"], schemas)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["model"], "test-model")
        handler = self.build.call_args.args[0]
        self.assertIsNone(handler.redirect_request(None, None, 302, "", {}, "https://other.example"))

    def test_429_retries_once_then_returns(self):
        self.opener.open.side_effect = [urllib.error.HTTPError("url", 429, "rate limit", {}, None),
            self.response(b'{"choices":[{"message":{"content":"ok"}}]}')]
        with patch("time.sleep") as sleep:
            self.assertEqual(self.client.complete([], []).kind, "final")
            self.assertEqual(self.opener.open.call_count, 2)
            sleep.assert_called_once()

    def test_auth_failure_not_retried_or_leaked(self):
        self.opener.open.side_effect = urllib.error.HTTPError("url", 401, "secret-test-key", {}, None)
        with self.assertRaises(LLMError) as caught:
            self.client.complete([], [])
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(self.opener.open.call_count, 1)

    def test_network_invalid_json_and_oversized_response(self):
        for value in (urllib.error.URLError("network"), self.response(b"not json"),
                      self.response(b"x" * 1000001)):
            with self.subTest(kind=type(value).__name__):
                self.opener.open.side_effect = value if isinstance(value, Exception) else None
                self.opener.open.return_value = value
                with self.assertRaises(LLMError):
                    self.client.complete([], [])


if __name__ == "__main__":
    unittest.main()
