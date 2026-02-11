#!/usr/bin/env bash
# Nahla — Droplet Startup Script (CPU-only, API-based)
# Idempotent — safe to run every time.
set -euo pipefail

PROJECT="/home/iman/cassie-project"
VENV="$PROJECT/venv"
LOG="$PROJECT/startup.log"

log() { echo "[startup] $*" | tee -a "$LOG"; }
log "Starting at $(date -Iseconds)"

# ── 1. Activate venv ─────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    log "Creating venv..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# ── 2. Ensure Python deps ────────────────────────────────────────
DEPS="qdrant-client sentence-transformers pyyaml mcp anthropic requests gradio langgraph langchain-core langchain-anthropic"
for pkg in qdrant-client sentence-transformers gradio langgraph anthropic; do
    python -c "import ${pkg//-/_}" 2>/dev/null || {
        log "Installing missing Python packages..."
        pip install -q $DEPS 2>&1 | tail -3
        break
    }
done

# ── 3. Ensure Qdrant binary ──────────────────────────────────────
if ! command -v qdrant &>/dev/null; then
    log "Installing Qdrant..."
    curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz \
        | tar xz -C /usr/local/bin/
fi

# ── 4. Start Qdrant (idempotent) ─────────────────────────────────
if curl -sf http://localhost:6333/collections >/dev/null 2>&1; then
    log "Qdrant already running"
else
    log "Starting Qdrant..."
    nohup qdrant --config-path "$PROJECT/memory/qdrant_config.yaml" \
        > "$PROJECT/memory/qdrant.log" 2>&1 &
    echo $! > "$PROJECT/memory/qdrant.pid"
    sleep 3
    if curl -sf http://localhost:6333/collections >/dev/null 2>&1; then
        log "Qdrant started OK"
    else
        log "WARNING: Qdrant failed to start — check memory/qdrant.log"
    fi
fi

# ── 5. Restore MEMORY.md (warp layer) if missing ─────────────────
MEMORY_DIR="$HOME/.claude/projects/home-iman-cassie--project/memory"
mkdir -p "$MEMORY_DIR"
if [ ! -f "$MEMORY_DIR/MEMORY.md" ]; then
    log "Restoring MEMORY.md from template..."
    cp "$PROJECT/memory/MEMORY_TEMPLATE.md" "$MEMORY_DIR/MEMORY.md"
fi

# ── 6. Register MCP voice-memory server if needed ────────────────
CLAUDE_CONFIG="$HOME/.claude.json"
if [ -f "$CLAUDE_CONFIG" ] && grep -q "voice-memory" "$CLAUDE_CONFIG" 2>/dev/null; then
    log "MCP voice-memory already registered"
else
    log "Registering MCP voice-memory server..."
    PYTHON_PATH="$VENV/bin/python3"
    MCP_SCRIPT="$PROJECT/memory/mcp_server.py"
    # Create or update claude config
    if [ ! -f "$CLAUDE_CONFIG" ]; then
        echo "{}" > "$CLAUDE_CONFIG"
    fi
    python3 -c "
import json
with open(\"$CLAUDE_CONFIG\") as f:
    config = json.load(f)
config.setdefault(\"mcpServers\", {})
config[\"mcpServers\"][\"voice-memory\"] = {
    \"command\": \"$PYTHON_PATH\",
    \"args\": [\"$MCP_SCRIPT\"],
    \"env\": {}
}
with open(\"$CLAUDE_CONFIG\", \"w\") as f:
    json.dump(config, f, indent=2)
"
    log "MCP voice-memory registered"
fi

log "Recovery complete. $(date -Iseconds)"

# ── Status ────────────────────────────────────────────────────────
QDRANT_STATUS="DOWN"
curl -sf http://localhost:6333/collections >/dev/null 2>&1 && QDRANT_STATUS="UP"
log "Services: Qdrant=$QDRANT_STATUS"
