# 验证记录

日期：2026-10-19（Asia/Shanghai）。

## 本地离线验证

环境：Windows、Python 3.12。

```text
python -m pip install -e .
python -m unittest discover -s tests -v
```

最终 **30 项离线测试全部通过**，耗时 16.569 秒。覆盖直接对话/追问、工具调用链、多调用结果配对、session/用户隔离、重启持久化、坏参数修正、轮次上限、API 故障、busy/租约、中断恢复、压缩与降级、mock 声明、计算器边界、协议解析、隐藏推理不保存及配置校验；另含连续 200 轮上下文预算检查、工具输出限制、待办删除隔离，以及 HTTP 请求格式、重定向、429 重试、401 脱敏、网络和响应异常测试。

`python -m compileall -q agent_harness tests scripts` 与 CLI `--help` 检查通过。`.env`、`.agent/` 与数据库被 Git 忽略。

GitHub Actions 配置 Python 3.11/3.12/3.13；远端结果需要以 Actions 实际运行状态为准，本地通过不等于远端已通过。

## 真实 LLM 验收

**入口已尝试启动，但因缺少 LLM_API_KEY 在发送 HTTP 请求前失败。** 当前没有成功的真实模型调用报告，因此“需要使用真实 LLM API”的端到端验收条件尚未闭环。配置失败也写入本地 `.agent/live-smoke.json`，不记为通过。

项目代码使用真实 HTTP API，运行时没有 mock LLM fallback；测试中的 ScriptedLLM 仅用于确定性验证。模型配置按用户要求使用 `gpt-6.1-sol`；其账号权限与网关兼容性未验证。

完成方法：在 `.env` 配置凭证，运行 `python scripts/live_smoke.py`，确认退出码为 0 且 `.agent/live-smoke.json` 中 `status=passed`。报告包含模型名、UTC 时间、检查项与演示回答，不含 API Key。分享前仍应检查演示输出内容。

不要在未执行时把这项改成通过，不要在 GitHub 提交密钥。
