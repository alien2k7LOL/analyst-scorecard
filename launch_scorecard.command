#!/bin/zsh
# Analyst Scorecard launcher.
# EASY ACCESS: double-click this file in Finder, or run  ./launch_scorecard.command
# It activates the project's virtualenv, starts the app, and opens it in your browser.
# Press Ctrl+C in the window to stop the app.

cd "$(dirname "$0")" || exit 1

# First-time setup: create the venv and install dependencies if they're missing.
if [ ! -x .venv/bin/streamlit ]; then
  echo "First-time setup — creating virtualenv and installing dependencies (one time only)…"
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt || {
    echo "Setup failed. Open an issue or run the commands in README.md manually."; exit 1; }
fi

PORT=8501
echo ""
echo "  Starting Analyst Scorecard at  http://localhost:${PORT}"
echo "  Your browser will open automatically. Press Ctrl+C here to stop."
echo ""

# Open the browser a moment after the server comes up.
( sleep 3; open "http://localhost:${PORT}" ) &

exec .venv/bin/streamlit run app.py \
  --server.headless true \
  --server.port "${PORT}" \
  --browser.gatherUsageStats false
