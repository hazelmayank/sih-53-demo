#!/usr/bin/env bash
# One-command launch (Linux / macOS / Git Bash):  ./start.sh
set -e
cd "$(dirname "$0")"

python -c "import fastapi, uvicorn, numpy, scipy, psutil, multipart" 2>/dev/null || python -m pip install -r requirements.txt

if [ ! -f frontend/dist/index.html ]; then
  (cd frontend && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm run build)
fi

echo "Dashboard: http://127.0.0.1:8053"
python -m backend --port 8053 --open
