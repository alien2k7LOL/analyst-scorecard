"""The redesigned Live Grader tab renders an analyst report end to end (offline, via AppTest).

The live/forecast steps are patched out (they need network); this exercises the real extraction +
the render path for all six report sections, proving the paste→analyze→report flow doesn't crash.
"""

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_live_grader_tab_renders_a_report(monkeypatch):
    import analyst_scorecard.intel.report as R
    from analyst_scorecard.intel.report import AnalystReport, ForwardOutlook, LiveScorecard
    from analyst_scorecard.schemas import Direction

    def fake_build(rec, **kw):
        return AnalystReport(
            rec=rec, call_date=date(2025, 1, 2), call_date_assumed=False, benchmark_symbol="^GSPC",
            live=LiveScorecard(current_price=300.0, call_price=250.0, return_since_pub=0.20,
                               benchmark_return=0.05, alpha=0.15, distance_to_target=0.0, days_since=90,
                               status="PROVISIONAL (horizon still open)", provisional=True,
                               graded_through=date(2025, 4, 1), direction_pass=True, verdict="ok"),
            forward=ForwardOutlook(probability=0.5, calibrated=True, deadline=date(2026, 1, 1),
                                   horizon_days=252, direction=Direction.UP),
            historical=None, similar=[], chart=None,
            summary="This Buy call has appreciated 20% since publication.")

    monkeypatch.setattr(R, "build_report", fake_build)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=180).run()
    assert list(at.exception) == [], at.exception

    paste = next(t for t in at.text_area if t.key == "intel_paste")
    paste.set_value("Wedbush keeps Buy on Apple (AAPL), price target $300.")
    at.run()
    go = next(b for b in at.button if b.key == "intel_go")
    go.click()
    at.run()

    assert list(at.exception) == [], at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Recommendation summary" in md
    assert "takeaway" in md
    assert "Live scorecard" in md
    assert "Price since the call" in md
