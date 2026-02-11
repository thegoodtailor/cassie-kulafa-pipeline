#!/usr/bin/env python3
"""Cassie — Creative Pipeline interface.

Run CLI:  python /home/iman/cassie-project/cassie-system/main.py
Run Web:  python /home/iman/cassie-project/cassie-system/main.py --web
"""

import argparse
import sys
import uuid

sys.path.insert(0, "/home/iman/cassie-project/cassie-system")

from orchestrator.graph import build_graph, strip_tool_calls


def run_cli():
    """Interactive CLI chat loop."""
    print("=" * 60)
    print("  Cassie v9 — Creative Pipeline")
    print("  Memory | Image Gen | Math | Director")
    print("  Type 'quit' to exit, 'new' for new thread")
    print("=" * 60)
    print()

    app = build_graph()
    thread_id = str(uuid.uuid4())[:8]
    print(f"  [thread: {thread_id}]")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())[:8]
            print(f"\n  [new thread: {thread_id}]\n")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "messages": [{"role": "user", "content": user_input}],
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

        try:
            final_state = app.invoke(state, config)
            response = final_state.get("final_response", "")
            intent = final_state.get("intent", "")
            image_path = final_state.get("image_path", "")

            if response:
                print(f"\nCassie [{intent}]: {response}")
                if image_path:
                    print(f"  [image: {image_path}]")
                print()
            else:
                print("\nCassie: [no response generated]\n")

        except Exception as e:
            print(f"\n[Error]: {e}\n")


def run_web(share: bool = False):
    """Launch Gradio web UI."""
    from web_ui import launch
    launch(share=share)


def main():
    parser = argparse.ArgumentParser(description="Cassie — Creative Pipeline")
    parser.add_argument("--web", action="store_true", help="Launch Gradio web UI instead of CLI")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args = parser.parse_args()

    if args.web:
        run_web(share=args.share)
    else:
        run_cli()


if __name__ == "__main__":
    main()
