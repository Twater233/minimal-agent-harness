import argparse
import json
import os
import sys

from .llm import OpenAICompatible, load_env
from .runtime import Agent
from .store import BusyError, Store


def main():
    parser = argparse.ArgumentParser(description="Persistent minimal agent (real LLM API).")
    parser.add_argument("--user", default="local")
    parser.add_argument("--session", default="main")
    parser.add_argument("--db")
    parser.add_argument("--message", help="Single-turn message; omit for interactive chat")
    parser.add_argument("--trace", action="store_true", help="Show last 100 session events and exit")
    parser.add_argument("--todos", action="store_true", help="Show current session todos and exit")
    args = parser.parse_args()
    load_env()
    store = Store(args.db or os.getenv("AGENT_DB", ".agent/state.db"))
    if args.trace or args.todos:
        value = store.events(args.user, args.session) if args.trace else store.todo(args.user, args.session, "list")
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    try:
        agent = Agent(store, OpenAICompatible.from_env())
        if args.message:
            print(agent.run(args.user, args.session, args.message))
            return
        print(f"Agent | user={args.user} session={args.session} | /exit 退出")
        while True:
            try:
                text = input("你 > ")
            except EOFError:
                break
            if text.strip() == "/exit":
                break
            if text.strip():
                try:
                    print("Agent >", agent.run(args.user, args.session, text))
                except (BusyError, ValueError) as exc:
                    print(str(exc), file=sys.stderr)
    except (ValueError, BusyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("\n已中断；待办和历史已保存。", file=sys.stderr)


if __name__ == "__main__":
    main()
