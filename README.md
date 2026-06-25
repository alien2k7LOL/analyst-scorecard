# Analyst Scorecard

> Honest, fair, reproducible grading of Wall Street analyst price targets.

Track analyst price targets, wait for each target's deadline, compare the call to what the
stock actually did, and grade each analyst honestly over time. The headline question:
**would you have done better just buying the index?**

This README is filled out fully in the final phase. Quick start:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                   # run the test suite
python -m analyst_scorecard.cli          # run the end-to-end simulation, print leaderboard
streamlit run app.py                     # launch the demo app
```

The engine is **offline-first**: it builds and runs with no network and no API key. The
Anthropic API is used only for optional call extraction and plain-English verdicts, both
behind interfaces with deterministic offline fallbacks.

See [`PLAN.md`](PLAN.md) for the architecture and phase plan, and
[`PROGRESS.md`](PROGRESS.md) for the running build log and every scoring assumption.
