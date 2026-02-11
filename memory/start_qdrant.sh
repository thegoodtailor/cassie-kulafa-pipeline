#!/bin/bash
# Ensure Qdrant server is running. Idempotent — safe to call multiple times.

QDRANT_PID_FILE="/workspace/memory/qdrant.pid"
QDRANT_LOG="/workspace/memory/qdrant.log"
QDRANT_CONFIG="/workspace/memory/qdrant_config.yaml"

# Check if already running
if [ -f "$QDRANT_PID_FILE" ]; then
    PID=$(cat "$QDRANT_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        # Verify it's actually responding
        if curl -s --max-time 2 http://localhost:6333/ >/dev/null 2>&1; then
            echo "Qdrant already running (PID $PID)"
            exit 0
        fi
    fi
    rm -f "$QDRANT_PID_FILE"
fi

# Start Qdrant
mkdir -p /workspace/memory/qdrant_data/storage /workspace/memory/qdrant_data/snapshots
nohup qdrant --config-path "$QDRANT_CONFIG" --disable-telemetry >> "$QDRANT_LOG" 2>&1 &
echo $! > "$QDRANT_PID_FILE"

# Wait for startup
for i in $(seq 1 10); do
    if curl -s --max-time 1 http://localhost:6333/ >/dev/null 2>&1; then
        echo "Qdrant started (PID $(cat $QDRANT_PID_FILE))"
        exit 0
    fi
    sleep 0.5
done

echo "WARNING: Qdrant may not have started properly. Check $QDRANT_LOG"
exit 1
