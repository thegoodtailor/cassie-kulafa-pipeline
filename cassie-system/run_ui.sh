#!/bin/bash
cd /home/iman/cassie-project/cassie-system
PYTHONUNBUFFERED=1 exec python3 -u -c "from web_ui import launch; launch(share=True)"
