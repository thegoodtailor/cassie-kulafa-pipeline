#!/usr/bin/env python3
"""Send a message to Cassie through the 004 config (GPT-5.1, temp 0.8, companion).
Usage: echo "message" | python experiments/nahla_send_004.py
"""
import os, sys

# Set 004 config before importing graph
os.environ["CASSIE_SYSTEM_PROMPT"] = "companion"
os.environ["CASSIE_DIRECTOR"] = "false"
os.environ["CASSIE_KITAB_RECALL"] = "false"
os.environ["CASSIE_MODEL"] = "gpt-5.1-chat-latest"
os.environ["CASSIE_TEMPERATURE"] = "1.0"  # GPT-5.1 only supports temperature=1

sys.path.insert(0, "/home/iman/cassie-project/cassie-system")
from dotenv import load_dotenv
load_dotenv("/home/iman/cassie-project/.env")
from orchestrator.graph import build_graph

msg = sys.stdin.read().strip()
if not msg:
    sys.exit(1)

app = build_graph()
state = {
    "messages": [{"role": "user", "content": msg}],
    "intent": "", "cassie_raw": "", "cassie_kitab_context": "",
    "cassie_conversation_context": "", "cassie_recall_decision": {},
    "director_output": {}, "image_path": "", "math_result": "",
    "final_response": "", "exchange_id": "", "tau_tgt": "",
}
result = app.invoke(state, {"configurable": {"thread_id": "nahla-cassie-004"}})
print(result.get("final_response", "[no response]"))
