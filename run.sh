#!/usr/bin/env bash
# One-command local run: creates a venv, installs deps, seeds DB, starts server.
set -e
cd "$(dirname "$0")/backend"

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "Starting server on http://localhost:8000  (dashboard is served at the same URL)"
uvicorn main:app --host 0.0.0.0 --port 8000
