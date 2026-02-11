#!/usr/bin/env python3
"""
Chat archive — saves conversation turns to both:
  1. A plain-text log file (human-readable archive)
  2. The Qdrant vector store (semantic retrieval)

Usage:
  # Archive a single exchange
  python3 -m memory.archive add --user "question" --assistant "response" --session "session-1"

  # Archive from Claude Code's jsonl conversation file
  python3 -m memory.archive ingest /path/to/conversation.jsonl --session "session-1"

  # List archived sessions
  python3 -m memory.archive sessions
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE_DIR = "/workspace/memory/chat_archive"
os.makedirs(ARCHIVE_DIR, exist_ok=True)


def archive_exchange(
    user_msg: str,
    assistant_msg: str,
    session_id: str = "",
    store_to_vector: bool = True,
):
    """Archive a user-assistant exchange."""
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    date_str = now.strftime("%Y-%m-%d")

    # 1. Append to daily log file
    log_path = os.path.join(ARCHIVE_DIR, f"{date_str}.md")
    with open(log_path, "a") as f:
        f.write(f"\n## [{timestamp}] {session_id}\n\n")
        f.write(f"**User:** {user_msg}\n\n")
        f.write(f"**Assistant:** {assistant_msg}\n\n")
        f.write("---\n")

    # 2. Store to vector store (assistant response as the searchable content)
    if store_to_vector:
        from memory.store import MemoryStore

        store = MemoryStore()

        # Archive the exchange as an event
        combined = f"User asked: {user_msg[:200]}\nResponse: {assistant_msg[:500]}"
        store.add(
            content=combined,
            entry_type="event",
            tags=["chat-archive"],
            source=f"archive/{session_id}",
            session_id=session_id,
        )
        store.close()

    return log_path


def ingest_jsonl(jsonl_path: str, session_id: str = ""):
    """Ingest a Claude Code conversation .jsonl file."""
    from memory.store import MemoryStore

    store = MemoryStore()
    path = Path(jsonl_path)

    if not path.exists():
        print(f"File not found: {jsonl_path}")
        sys.exit(1)

    exchanges = []
    current_user = None

    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Claude Code jsonl format: top-level "type" is "user" or "assistant",
            # actual content is in entry["message"]["content"] (string or list of blocks)
            entry_type = entry.get("type", "")
            message = entry.get("message", {})
            role = message.get("role", "") or entry_type
            raw_content = message.get("content", "")

            content = ""
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                content = " ".join(text_parts)

            if not content.strip():
                continue

            if role == "user":
                current_user = content
            elif role == "assistant" and current_user:
                exchanges.append((current_user, content))
                current_user = None

    # Archive each exchange
    count = 0
    for user_msg, assistant_msg in exchanges:
        # Skip very short exchanges (tool calls, etc.)
        if len(user_msg) < 10 and len(assistant_msg) < 10:
            continue

        archive_exchange(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            session_id=session_id,
            store_to_vector=True,
        )
        count += 1

    store.close()
    print(f"Archived {count} exchanges from {jsonl_path}")
    return count


def list_sessions():
    """List archived session files."""
    archive_path = Path(ARCHIVE_DIR)
    files = sorted(archive_path.glob("*.md"))
    if not files:
        print("No archived sessions found.")
        return

    for f in files:
        size = f.stat().st_size
        print(f"  {f.name}  ({size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Chat archive manager")
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    add_p = sub.add_parser("add", help="Archive a single exchange")
    add_p.add_argument("--user", required=True, help="User message")
    add_p.add_argument("--assistant", required=True, help="Assistant response")
    add_p.add_argument("--session", default="")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest a Claude Code .jsonl conversation")
    ingest_p.add_argument("path", help="Path to .jsonl file")
    ingest_p.add_argument("--session", default="")

    # sessions
    sub.add_parser("sessions", help="List archived sessions")

    args = parser.parse_args()

    if args.command == "add":
        path = archive_exchange(args.user, args.assistant, args.session)
        print(f"Archived to {path}")
    elif args.command == "ingest":
        ingest_jsonl(args.path, args.session)
    elif args.command == "sessions":
        list_sessions()


if __name__ == "__main__":
    main()
