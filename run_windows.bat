@echo off
cd /d "%~dp0backend"

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

echo Starting server on http://localhost:8000  (dashboard is served at the same URL)
uvicorn main:app --host 0.0.0.0 --port 8000
