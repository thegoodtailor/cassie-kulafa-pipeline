"""Thread persistence — shared by CLI and web frontends.

JSON-per-thread on disk. Both cli.py and web_app.py import from here.
"""

import ast
import json
import os
import uuid
from datetime import datetime


HISTORY_DIR = "/home/iman/cassie-project/cassie-system/data/chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)


def history_path(thread_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{thread_id}.json")


def save_history(thread_id: str, history: list):
    try:
        with open(history_path(thread_id), "w") as f:
            json.dump(history, f)
    except Exception:
        pass


def extract_preview_text(content: str) -> str:
    """Unwrap legacy Gradio format strings like [{'text': '...'}] up to 10 levels deep."""
    if not isinstance(content, str):
        return str(content)[:80]
    for _ in range(10):
        stripped = content.strip()
        if stripped.startswith("[{") and "'text'" in stripped:
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    content = parsed[0].get("text", stripped)
                else:
                    break
            except (ValueError, SyntaxError):
                break
        else:
            break
    return content.strip()


def load_history(thread_id: str) -> list:
    """Load and normalize thread history (handles legacy Gradio format)."""
    path = history_path(thread_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        normalized = []
        for msg in data:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Legacy image format from Gradio UI
            if msg.get("_type") == "image":
                img_path = content
                if os.path.isfile(img_path):
                    normalized.append({
                        "role": role,
                        "content": "",
                        "image": f"/images/{os.path.basename(img_path)}",
                    })
                continue

            # Legacy Gradio format
            if isinstance(content, str) and content.startswith("[{") and "'text'" in content:
                try:
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, list) and parsed:
                        content = parsed[0].get("text", content)
                except (ValueError, SyntaxError):
                    pass

            # Normalize image paths: filesystem -> web URL
            image_val = msg.get("image")
            if image_val and not image_val.startswith("/images/"):
                if os.path.isfile(image_val):
                    image_val = f"/images/{os.path.basename(image_val)}"
                else:
                    image_val = None

            normalized.append({
                "role": role,
                "content": content,
                **({"image": image_val} if image_val else {}),
            })
        return normalized
    except Exception:
        return []


def list_threads() -> list[dict]:
    threads = []
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith(".json") and not fname.startswith("_"):
            tid = fname[:-5]
            path = os.path.join(HISTORY_DIR, fname)
            try:
                mtime = os.path.getmtime(path)
                with open(path) as f:
                    data = json.load(f)
                preview = ""
                msg_count = 0
                for msg in data:
                    if isinstance(msg, dict):
                        msg_count += 1
                        if msg.get("role") == "user" and not preview:
                            raw = str(msg.get("content", ""))
                            preview = extract_preview_text(raw)[:80]
                if not preview:
                    preview = "(empty)"
                threads.append({
                    "id": tid,
                    "preview": preview,
                    "timestamp": datetime.fromtimestamp(mtime).isoformat(),
                    "message_count": msg_count,
                })
            except Exception:
                threads.append({"id": tid, "preview": tid, "timestamp": "", "message_count": 0})
    threads.sort(key=lambda x: x["timestamp"], reverse=True)
    return threads


def save_message(thread_id: str, role: str, content: str, image_path: str = ""):
    history = load_history(thread_id)
    if image_path:
        image_url = f"/images/{os.path.basename(image_path)}" if not image_path.startswith("/images/") else image_path
        history.append({"role": role, "content": content, "image": image_url})
    else:
        history.append({"role": role, "content": content})
    save_history(thread_id, history)


def create_thread(name: str = "") -> str:
    """Create a new empty thread. Returns thread_id."""
    tid = name or str(uuid.uuid4())[:8]
    save_history(tid, [])
    return tid
