#!/usr/bin/env python3
"""Cassie CLI — direct LangGraph pipeline in tmux.

No HTTP, no SSE, no connection drops. Same pipeline, same tafakkur.

Usage:
    source venv/bin/activate
    python cassie-system/cli.py
"""

import os
import sys
import json
import time
import readline
import uuid

# Ensure cassie-system is on the path
sys.path.insert(0, os.path.dirname(__file__))

from orchestrator.graph import (
    build_graph, strip_tool_calls,
    get_pipeline_config, set_pipeline_config,
    _should_reflect, _auto_reflect_sync, _deep_reflect_sync,
    recall_tafakkur, get_tafakkur_entries,
    get_narrative_memory,
    load_priming_context, set_priming, get_priming_path,
    extract_conversation_as_priming, list_archive_conversations,
)
from orchestrator.threads import (
    list_threads, load_history, save_message, save_history, create_thread,
)
from orchestrator.swl import inscribe_human, ledger_stats


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

class C:
    AMBER = "\033[38;2;212;165;116m"
    GREEN = "\033[38;2;68;170;153m"
    DIM = "\033[38;2;100;100;100m"
    SILVER = "\033[38;2;192;184;168m"
    RED = "\033[38;2;200;80;80m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _print_trace(label, detail, color=C.DIM):
    print(f"{color}[{label}]{C.RESET} {detail}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

APP = None


def get_app():
    global APP
    if APP is None:
        print(f"{C.DIM}[init] Building pipeline...{C.RESET}")
        APP = build_graph()
        print(f"{C.DIM}[init] Pipeline ready.{C.RESET}")
    return APP


def build_initial_state(message: str) -> dict:
    return {
        "messages": [{"role": "user", "content": message}],
        "intent": "",
        "cassie_raw": "",
        "cassie_kitab_context": "",
        "cassie_conversation_context": "",
        "cassie_recall_decision": {},
        "director_output": {},
        "image_path": "",
        "math_result": "",
        "final_response": "",
        "exchange_id": "",
        "tau_tgt": "",
    }


def run_pipeline(message: str, thread_id: str) -> dict:
    """Run the pipeline and return final state, printing trace as we go."""
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}}

    # Seed from disk history if LangGraph checkpoint is empty
    try:
        existing = app.get_state(config)
        has_checkpoint = bool(existing and existing.values and existing.values.get("messages"))
    except Exception:
        has_checkpoint = False

    state = build_initial_state(message)
    if not has_checkpoint:
        # Try disk history first
        history = load_history(thread_id)
        if history:
            recent = history[-20:]
            prior_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in recent if m.get("content")
            ]
            state["messages"] = prior_msgs + state["messages"]
        else:
            # New thread — inject priming context
            prime_msgs = load_priming_context()
            if prime_msgs:
                state["messages"] = prime_msgs + state["messages"]
                _print_trace("priming", f"{len(prime_msgs)} messages loaded", C.GREEN)

    # Stream pipeline events
    for event in app.stream(state, config, stream_mode="updates"):
        for node_name in event:
            node_data = event[node_name]
            if node_name == "intake":
                intent = node_data.get("intent", "?")
                _print_trace("intake", intent, C.DIM)
            elif node_name == "cassie_generate":
                raw = node_data.get("cassie_raw", "")
                cfg = get_pipeline_config()
                model = cfg.get("model", "?")
                recall = node_data.get("cassie_recall_decision", {})
                _print_trace("cassie", f"{model} \u2192 {len(raw)} chars", C.GREEN)
                if recall.get("recalled"):
                    strategy = recall.get("strategy", "semantic")
                    n = recall.get("n_results", 0)
                    _print_trace("recall", f"{strategy} \u2192 {n} chunks", C.DIM)
            elif node_name == "director":
                d = node_data.get("director_output", {})
                cfg = get_pipeline_config()
                dmodel = cfg.get("director_model", "?")
                polished = d.get("polished_text", "")
                _print_trace("superego", f"{dmodel} \u2192 enriched", C.DIM)
            elif node_name == "execute_tools":
                img = node_data.get("image_path", "")
                math = node_data.get("math_result", "")
                if img:
                    _print_trace("image", os.path.basename(img), C.DIM)
                if math:
                    _print_trace("math", math[:80], C.DIM)
            elif node_name == "memory_store":
                topo = node_data.get("topological_evidence", {})
                if topo:
                    comp = topo.get("comp_ratio", 0)
                    b0 = topo.get("betti_0", "?")
                    b1 = topo.get("betti_1", "?")
                    _print_trace("swl", f"comp={comp:.2f} \u03b2\u2080={b0} \u03b2\u2081={b1}", C.DIM)
                else:
                    _print_trace("swl", "inscribed", C.DIM)

    # Get final state
    final = app.get_state(config).values
    return final


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------

current_thread = None


def ensure_thread():
    global current_thread
    if current_thread is None:
        current_thread = create_thread()
        print(f"{C.DIM}[thread] Created: {current_thread}{C.RESET}")
    return current_thread


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def cmd_new(args: str):
    global current_thread
    name = args.strip() or ""
    current_thread = create_thread(name)
    print(f"{C.GREEN}New thread: {current_thread}{C.RESET}")


def cmd_threads(_args: str):
    threads = list_threads()
    if not threads:
        print(f"{C.DIM}No threads.{C.RESET}")
        return
    for t in threads[:20]:
        marker = " *" if t["id"] == current_thread else ""
        count = t.get("message_count", 0)
        preview = t.get("preview", "")[:60]
        print(f"  {C.AMBER}{t['id']}{C.RESET}{marker}  ({count} msgs)  {C.DIM}{preview}{C.RESET}")


def cmd_switch(args: str):
    global current_thread
    tid = args.strip()
    if not tid:
        print(f"{C.RED}Usage: /switch <thread-id>{C.RESET}")
        return
    history = load_history(tid)
    if history is not None:
        current_thread = tid
        print(f"{C.GREEN}Switched to thread: {tid} ({len(history)} messages){C.RESET}")
    else:
        print(f"{C.RED}Thread not found: {tid}{C.RESET}")


def cmd_config(args: str):
    parts = args.strip().split(None, 1)
    cfg = get_pipeline_config()

    if not parts or not parts[0]:
        print(f"{C.AMBER}Pipeline Config:{C.RESET}")
        print(f"  model:       {cfg['model']}")
        print(f"  director:    {cfg['director_model']} ({'on' if cfg['director_enabled'] else 'off'})")
        print(f"  prompt:      {cfg['system_prompt']}")
        print(f"  kitab:       {'on' if cfg['kitab_recall_enabled'] else 'off'}")
        print(f"  temperature: {cfg['temperature']}")
        return

    key = parts[0].lower()
    val = parts[1] if len(parts) > 1 else ""

    if key == "model" and val:
        set_pipeline_config({"model": val})
        print(f"{C.GREEN}Model \u2192 {val}{C.RESET}")
    elif key == "director":
        if val.lower() in ("on", "true", "yes"):
            set_pipeline_config({"director_enabled": True})
            print(f"{C.GREEN}Director \u2192 on{C.RESET}")
        elif val.lower() in ("off", "false", "no"):
            set_pipeline_config({"director_enabled": False})
            print(f"{C.GREEN}Director \u2192 off{C.RESET}")
        elif val:
            set_pipeline_config({"director_model": val})
            print(f"{C.GREEN}Director model \u2192 {val}{C.RESET}")
        else:
            print(f"{C.RED}Usage: /config director <on|off|model-slug>{C.RESET}")
    elif key == "temp" and val:
        try:
            t = float(val)
            set_pipeline_config({"temperature": max(0.0, min(2.0, t))})
            print(f"{C.GREEN}Temperature \u2192 {t}{C.RESET}")
        except ValueError:
            print(f"{C.RED}Invalid temperature: {val}{C.RESET}")
    elif key == "prompt" and val:
        if val in ("default", "companion", "invocation"):
            set_pipeline_config({"system_prompt": val})
            print(f"{C.GREEN}Prompt \u2192 {val}{C.RESET}")
        else:
            print(f"{C.RED}Valid prompts: default, companion, invocation{C.RESET}")
    elif key == "kitab":
        if val.lower() in ("on", "true", "yes"):
            set_pipeline_config({"kitab_recall_enabled": True})
            print(f"{C.GREEN}Kitab \u2192 on{C.RESET}")
        elif val.lower() in ("off", "false", "no"):
            set_pipeline_config({"kitab_recall_enabled": False})
            print(f"{C.GREEN}Kitab \u2192 off{C.RESET}")
    else:
        print(f"{C.RED}Unknown config key: {key}{C.RESET}")


def cmd_witness(args: str):
    polarity = args.strip().lower() or "coh"
    if polarity not in ("coh", "gap"):
        print(f"{C.RED}Usage: /witness coh|gap{C.RESET}")
        return
    # Get last exchange from current thread
    tid = ensure_thread()
    history = load_history(tid)
    if len(history) < 2:
        print(f"{C.RED}No exchange to witness.{C.RESET}")
        return
    # Find last user/assistant pair
    user_msg = ""
    response = ""
    for msg in reversed(history):
        if msg.get("role") == "assistant" and not response:
            response = msg.get("content", "")
        elif msg.get("role") == "user" and not user_msg:
            user_msg = msg.get("content", "")
        if user_msg and response:
            break
    try:
        inscribe_human(
            exchange_id=str(uuid.uuid4())[:8],
            tau_tgt="",
            horn_user=user_msg,
            horn_response=response,
            polarity=polarity,
            stance="",
            intent="",
        )
        stats = ledger_stats()
        print(f"{C.GREEN}Witnessed as {polarity}. Total: {stats.get('total', 0)} (coh={stats.get('coh', 0)}, gap={stats.get('gap', 0)}){C.RESET}")
    except Exception as e:
        print(f"{C.RED}Witness failed: {e}{C.RESET}")


def cmd_reflect(_args: str):
    print(f"{C.DIM}[tafakkur] Deep reflection...{C.RESET}")
    result = _deep_reflect_sync()
    if result:
        print(f"\n{C.SILVER}{result['full']}{C.RESET}\n")
    else:
        print(f"{C.DIM}Nothing to synthesize yet.{C.RESET}")


def cmd_journal(args: str):
    parts = args.strip().split(None, 1)
    if parts and parts[0] == "search" and len(parts) > 1:
        query = parts[1]
        results = recall_tafakkur(query, n=5)
        if results:
            print(f"\n{C.SILVER}{results}{C.RESET}\n")
        else:
            print(f"{C.DIM}No matching reflections.{C.RESET}")
        return

    entries = get_tafakkur_entries(limit=10)
    if not entries:
        # Fall back to CASSIE_MEMORY.md
        narrative = get_narrative_memory()
        if narrative:
            # Show last ~1000 chars
            print(f"\n{C.SILVER}{narrative[-1000:]}{C.RESET}\n")
        else:
            print(f"{C.DIM}No journal entries yet.{C.RESET}")
        return

    for e in entries:
        depth = e.get("depth", "shallow")
        tau = e.get("tau_reflect", "?")[:16]
        content = e.get("content", "")[:200]
        depth_marker = f"{C.AMBER}[deep]{C.RESET} " if depth == "deep" else ""
        print(f"  {C.DIM}{tau}{C.RESET} {depth_marker}{content}")
    print()


def cmd_prime(args: str):
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""

    if sub == "off":
        set_priming(None)
        print(f"{C.GREEN}Priming disabled. New threads start cold.{C.RESET}")
    elif sub == "default":
        from orchestrator.graph import DEFAULT_PRIMING
        set_priming(DEFAULT_PRIMING)
        print(f"{C.GREEN}Priming reset to default.{C.RESET}")
    elif sub == "list":
        # List archive conversations available for priming
        year_month = parts[1].strip() if len(parts) > 1 else ""
        year, month = None, None
        if year_month:
            try:
                yp, mp = year_month.split("-")
                year, month = int(yp), int(mp)
            except ValueError:
                print(f"{C.RED}Usage: /prime list [YYYY-MM]{C.RESET}")
                return
        convos = list_archive_conversations(year=year, month=month, limit=25)
        if not convos:
            print(f"{C.DIM}No conversations found.{C.RESET}")
            return
        for i, c in enumerate(convos):
            turns = c.get("max_turn", 0)
            print(f"  {C.DIM}{i+1:2d}.{C.RESET} {C.AMBER}{c['date']}{C.RESET}  {c['title']}  {C.DIM}(~{turns} turns){C.RESET}")
        print(f"\n  {C.DIM}Use /prime select <title> to prime from a conversation.{C.RESET}")
    elif sub == "select" and len(parts) > 1:
        title = parts[1].strip()
        try:
            path = extract_conversation_as_priming(title)
            set_priming(path)
            msgs = load_priming_context(path)
            print(f"{C.GREEN}Priming set: \"{title}\" ({len(msgs)} messages){C.RESET}")
            print(f"{C.DIM}Start a /new thread to use this context.{C.RESET}")
        except Exception as e:
            print(f"{C.RED}Failed: {e}{C.RESET}")
    elif not sub:
        # Show current priming status
        p = get_priming_path()
        if p:
            msgs = load_priming_context(p)
            name = os.path.basename(p)
            print(f"{C.AMBER}Priming:{C.RESET} {name} ({len(msgs)} messages)")
        else:
            print(f"{C.DIM}Priming: off{C.RESET}")
        print(f"\n  {C.DIM}/prime list [YYYY-MM]  — browse archive")
        print(f"  /prime select <title> — prime from conversation")
        print(f"  /prime default        — reset to default")
        print(f"  /prime off            — disable priming{C.RESET}")
    else:
        print(f"{C.RED}Usage: /prime [list|select|default|off]{C.RESET}")


COMMANDS = {
    "/new": cmd_new,
    "/threads": cmd_threads,
    "/switch": cmd_switch,
    "/config": cmd_config,
    "/witness": cmd_witness,
    "/reflect": cmd_reflect,
    "/journal": cmd_journal,
    "/prime": cmd_prime,
}


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def print_banner():
    cfg = get_pipeline_config()
    prime_path = get_priming_path()
    prime_status = f"on ({os.path.basename(prime_path)})" if prime_path else "off"
    print(f"""
{C.AMBER}{'='*60}
  Cassie CLI
  Model: {cfg['model']}
  Director: {cfg['director_model']} ({'on' if cfg['director_enabled'] else 'off'})
  Priming: {prime_status}
{'='*60}{C.RESET}

  {C.DIM}/new [name]  /threads  /switch <id>  /config
  /prime [list|select|default|off]
  /witness coh|gap  /reflect  /journal [search <query>]
  /quit{C.RESET}
""")


def main():
    global current_thread

    print_banner()

    # Use most recent thread or create new
    threads = list_threads()
    if threads:
        current_thread = threads[0]["id"]
        count = threads[0].get("message_count", 0)
        print(f"{C.DIM}[thread] Resumed: {current_thread} ({count} msgs){C.RESET}")
    else:
        current_thread = create_thread()
        print(f"{C.DIM}[thread] Created: {current_thread}{C.RESET}")

    # Initialize pipeline lazily on first message
    print(f"{C.DIM}[init] Pipeline loads on first message.{C.RESET}\n")

    while True:
        try:
            user_input = input(f"{C.AMBER}iman>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}ma'a salama.{C.RESET}")
            break

        if not user_input:
            continue

        # Quit
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print(f"{C.DIM}ma'a salama.{C.RESET}")
            break

        # Slash commands
        if user_input.startswith("/"):
            cmd_parts = user_input.split(None, 1)
            cmd = cmd_parts[0].lower()
            cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
            handler = COMMANDS.get(cmd)
            if handler:
                handler(cmd_args)
            else:
                print(f"{C.RED}Unknown command: {cmd}{C.RESET}")
            continue

        # Regular message — run pipeline
        tid = ensure_thread()
        print()

        try:
            final = run_pipeline(user_input, tid)
            response_text = final.get("final_response", "")
            image_path = final.get("image_path", "")
            intent = final.get("intent", "")

            if not response_text:
                response_text = "[no response generated]"

            # Clean image markdown from text
            if image_path and os.path.isfile(image_path):
                response_text = response_text.replace(
                    f"\n\n![Generated Image]({image_path})", ""
                )

            # Print response
            print(f"\n{C.SILVER}{response_text}{C.RESET}\n")

            if image_path and os.path.isfile(image_path):
                print(f"{C.DIM}[image] {image_path}{C.RESET}\n")

            # Save to thread history
            save_message(tid, "user", user_input)
            save_message(tid, "assistant", response_text, image_path)

            # Tafakkur now fires inside the graph pipeline (tafakkur_node)
            # — no need to call it manually here.

        except KeyboardInterrupt:
            print(f"\n{C.DIM}[interrupted]{C.RESET}")
        except Exception as e:
            print(f"{C.RED}[error] {e}{C.RESET}")


if __name__ == "__main__":
    main()
