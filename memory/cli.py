#!/usr/bin/env python3
"""
CLI for the voice memory store.

Usage:
  python -m memory.cli add "content" --type insight --tags tag1,tag2 --source "session"
  python -m memory.cli search "query" --limit 5 --type insight
  python -m memory.cli list [--type insight] [--limit 20]
  python -m memory.cli count
"""

import argparse
import json
import sys

from memory.store import MemoryStore, ENTRY_TYPES


def main():
    parser = argparse.ArgumentParser(description="Voice memory store CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    add_p = sub.add_parser("add", help="Add a memory entry")
    add_p.add_argument("content", help="The memory content text")
    add_p.add_argument("--type", choices=ENTRY_TYPES, default="insight")
    add_p.add_argument("--tags", default="", help="Comma-separated tags")
    add_p.add_argument("--source", default="")
    add_p.add_argument("--session", default="")

    # search
    search_p = sub.add_parser("search", help="Semantic search")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--limit", type=int, default=5)
    search_p.add_argument("--type", choices=ENTRY_TYPES, default=None)
    search_p.add_argument("--tag", default=None)
    search_p.add_argument("--threshold", type=float, default=0.0)

    # list
    list_p = sub.add_parser("list", help="List all memories")
    list_p.add_argument("--type", choices=ENTRY_TYPES, default=None)
    list_p.add_argument("--limit", type=int, default=100)

    # count
    sub.add_parser("count", help="Count total memories")

    args = parser.parse_args()

    store = MemoryStore()

    try:
        if args.command == "add":
            tags = [t.strip() for t in args.tags.split(",") if t.strip()]
            pid = store.add(
                content=args.content,
                entry_type=args.type,
                tags=tags,
                source=args.source,
                session_id=args.session,
            )
            print(f"Added: {pid}")

        elif args.command == "search":
            results = store.search(
                query=args.query,
                limit=args.limit,
                entry_type=args.type,
                tag=args.tag,
                score_threshold=args.threshold,
            )
            for r in results:
                score = r.pop("score", 0)
                print(f"\n--- [{score:.3f}] {r.get('entry_type', '?')} ---")
                print(r.get("content", ""))
                if r.get("tags"):
                    print(f"  tags: {r['tags']}")
                if r.get("source"):
                    print(f"  source: {r['source']}")
                print(f"  created: {r.get('created_at', '?')}")

        elif args.command == "list":
            results = store.get_all(entry_type=args.type, limit=args.limit)
            for r in results:
                print(f"\n--- {r.get('entry_type', '?')} ---")
                content = r.get("content", "")
                print(content[:200] + ("..." if len(content) > 200 else ""))
                if r.get("tags"):
                    print(f"  tags: {r['tags']}")
                print(f"  created: {r.get('created_at', '?')}")

        elif args.command == "count":
            print(f"Total memories: {store.count()}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
