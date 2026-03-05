#!/bin/bash
# Cassie pipeline watchdog — checks health endpoint, restarts if unresponsive
# Called by cassie-watchdog.timer every 2 minutes

HEALTH_URL="http://127.0.0.1:7860/api/health"
SERVICE="cassie-pipeline"
LOG_TAG="cassie-watchdog"

response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$HEALTH_URL" 2>/dev/null)

if [ "$response" = "200" ]; then
    # All good — silent success
    exit 0
else
    logger -t "$LOG_TAG" "Health check failed (HTTP $response). Restarting $SERVICE..."
    systemctl restart "$SERVICE"
    logger -t "$LOG_TAG" "Restart issued."
fi
