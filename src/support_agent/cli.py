"""Command line: seed the demo database, chat in the terminal, list tools, or serve HTTP."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .agent import StopReason, SupportAgent, TurnResult
from .config import Settings
from .db import CUSTOMERS, ORDERS, open_demo_db
from .redact import RedactingFilter
from .tools import Toolbox


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, handlers=[handler])


def cmd_seed(args: argparse.Namespace, settings: Settings) -> int:
    path = Path(settings.db_path)
    if args.reset and path.exists():
        path.unlink()
    open_demo_db(path).close()
    print(f"Demo database ready at {path}")
    for customer_id, name, *_ in CUSTOMERS:
        orders = [o[0] for o in ORDERS if o[1] == customer_id]
        print(f"  {customer_id} {name}: {', '.join(orders)}")
    return 0


def cmd_tools(args: argparse.Namespace, settings: Settings) -> int:
    print(json.dumps(Toolbox().definitions(), indent=2))
    return 0


def _print_turn(result: TurnResult) -> None:
    print(f"agent> {result.reply}")
    if result.tool_calls:
        print(f"       (tools: {', '.join(result.tool_calls)}; stop: {result.stop_reason})")


def _review(agent: SupportAgent, result: TurnResult, reviewer: str) -> TurnResult:
    """Stand-in for a reviewer UI: ask in the same terminal."""
    while result.stop_reason is StopReason.AWAITING_APPROVAL and result.pending_approvals:
        pending = result.pending_approvals[0]
        answer = input(
            f"[reviewer {reviewer}] approve refund of {pending['amount']} for "
            f"{pending['order_id']} (reason: {pending['reason']})? [y/N] "
        )
        result = agent.decide(
            pending["approval_id"],
            approve=answer.strip().lower().startswith("y"),
            reviewer=reviewer,
        )
        _print_turn(result)
    return result


def cmd_chat(args: argparse.Namespace, settings: Settings) -> int:
    import anthropic

    from .factory import build_agent
    from .model import ModelError

    agent = build_agent(settings)
    conv = agent.start(args.customer)
    print(f"Chatting as {args.customer} with {settings.model}. Type /quit to leave.")
    print(f"Trace: {Path(settings.trace_dir) / (conv.id + '.jsonl')}")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            return 0
        try:
            result = agent.send(conv, text)
            _print_turn(result)
            _review(agent, result, args.reviewer)
        except ModelError as exc:
            if isinstance(exc.__cause__, anthropic.AuthenticationError):
                print("Authentication failed: set ANTHROPIC_API_KEY in the environment.")
                return 2
            print(f"Model API error: {exc}")


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .api import create_app
    from .factory import build_agent

    uvicorn.run(create_app(lambda: build_agent(settings)), host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-agent", description=__doc__)
    parser.add_argument("--db", help="SQLite path (env SUPPORT_AGENT_DB)")
    parser.add_argument("--trace-dir", help="JSONL trace directory (env SUPPORT_AGENT_TRACE_DIR)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed", help="create the demo database")
    seed.add_argument("--reset", action="store_true", help="delete and recreate it")
    seed.set_defaults(func=cmd_seed)

    tools = sub.add_parser("tools", help="print the tool definitions sent to the model")
    tools.set_defaults(func=cmd_tools)

    chat = sub.add_parser("chat", help="chat in the terminal (needs ANTHROPIC_API_KEY)")
    chat.add_argument("--customer", default="C-1001", help="signed-in demo customer id")
    chat.add_argument("--reviewer", default="demo-reviewer", help="name recorded on approvals")
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", help="run the HTTP API with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    settings = Settings.from_env()
    if args.db:
        settings = replace(settings, db_path=args.db)
    if args.trace_dir:
        settings = replace(settings, trace_dir=args.trace_dir)
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
