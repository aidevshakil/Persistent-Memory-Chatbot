"""Command-line inspector for the memory store.

Useful for proving persistence without the UI, and for scripted demos:

    python -m memory_chatbot.cli list
    python -m memory_chatbot.cli chat "I am 26 now"
    python -m memory_chatbot.cli recall "what sport should I try"
    python -m memory_chatbot.cli export --out memories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memory_chatbot.chat.engine import DEFAULT_USER_ID, build_chat_engine
from memory_chatbot.config import get_settings
from memory_chatbot.logging_utils import configure_logging
from memory_chatbot.models import MemoryCategory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memcli",
        description="Inspect and exercise the persistent memory store.",
    )
    parser.add_argument("-u", "--user", default=DEFAULT_USER_ID, help="memory profile id")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress log output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show the active configuration and storage paths")
    sub.add_parser("list", help="print every stored fact, grouped by category")

    recall = sub.add_parser("recall", help="show what would be injected for a query")
    recall.add_argument("query", nargs="+")
    recall.add_argument("-k", "--limit", type=int, default=None)

    chat = sub.add_parser("chat", help="run one full turn (recall + reply + remember)")
    chat.add_argument("message", nargs="+")

    remember = sub.add_parser("remember", help="store a fact directly")
    remember.add_argument("key")
    remember.add_argument("value")
    remember.add_argument(
        "-c", "--category", default=str(MemoryCategory.OTHER),
        choices=[str(c) for c in MemoryCategory],
    )
    remember.add_argument("-i", "--importance", type=float, default=0.7)

    forget = sub.add_parser("forget", help="delete one memory by id, or all with --all")
    forget.add_argument("memory_id", nargs="?")
    forget.add_argument("--all", action="store_true", help="delete every memory for the user")

    export = sub.add_parser("export", help="dump memories as JSON")
    export.add_argument("-o", "--out", type=Path, help="write to a file instead of stdout")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging("ERROR" if args.quiet else None)
    # Stored values may contain characters a legacy Windows console codepage
    # cannot encode; replace them rather than dying with UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    settings = get_settings()
    engine = build_chat_engine(settings)
    user = args.user

    try:
        match args.command:
            case "info":
                print(f"provider        : {engine.provider_name}")
                print(f"model           : {engine.model_name}")
                print(f"memory backend  : {engine.backend_name}")
                print(f"storage         : {engine.memory.storage_location}")
                print(f"data dir        : {settings.data_dir}")
                print(f"embeddings      : {settings.embedding_provider}")
                print(f"top_k / min     : {settings.memory_top_k} / {settings.memory_min_score}")
                print(f"user '{user}'   : {engine.memory.count(user)} memories")

            case "list":
                grouped = engine.memory.profile_by_category(user)
                if not grouped:
                    print(f"No memories stored for '{user}'.")
                    return 0
                for category, records in grouped.items():
                    print(f"\n[{category}]")
                    for record in records:
                        past = f"  (was: {record.history[0]})" if record.history else ""
                        print(
                            f"  {record.key:<24} {record.value}"
                            f"{past}  [imp {record.importance:.2f} | "
                            f"{record.updated_at:%Y-%m-%d %H:%M} | {record.id[:8]}]"
                        )

            case "recall":
                query = " ".join(args.query)
                results = engine.memory.recall(user, query, limit=args.limit)
                if not results:
                    print("Nothing relevant recalled.")
                    return 0
                print(f"Recalled {len(results)} memories for {query!r}:")
                for item in results:
                    print(f"  {item.score:.3f}  [{item.record.category}] {item.record.as_sentence()}")

            case "chat":
                message = " ".join(args.message)
                result = engine.respond(user, message)
                print(f"\nyou > {message}")
                if result.recalled:
                    print("recalled:")
                    for item in result.recalled:
                        print(f"  {item.score:.3f}  {item.record.as_sentence()}")
                print(f"\nbot > {result.reply.content}\n")
                if writes := result.persisted_writes:
                    print("memory writes:")
                    for write in writes:
                        print(f"  {write.operation.value:<7} {write.describe()[2:]}")

            case "remember":
                write = engine.memory.remember_fact(
                    user,
                    category=args.category,
                    key=args.key,
                    value=args.value,
                    importance=args.importance,
                )
                print(f"{write.operation.value}: {write.describe()[2:]}")

            case "forget":
                if args.all:
                    print(f"Deleted {engine.memory.forget_all(user)} memories for '{user}'.")
                elif args.memory_id:
                    ok = engine.memory.forget(args.memory_id)
                    print("Deleted." if ok else "No memory with that id.")
                else:
                    print("Provide a memory id or --all.", file=sys.stderr)
                    return 2

            case "export":
                payload = json.dumps(engine.memory.export(user), indent=2, ensure_ascii=False)
                if args.out:
                    args.out.write_text(payload, encoding="utf-8")
                    print(f"Wrote {args.out}")
                else:
                    print(payload)
    finally:
        engine.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
