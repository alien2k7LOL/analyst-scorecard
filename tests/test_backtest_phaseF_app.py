"""Back-test Phase F — the Streamlit app surfaces the historical back-test."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_app_renders_historical_tab():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=180).run()
    assert list(at.exception) == [], at.exception

    headers = [h.value for h in at.header]
    subheaders = [h.value for h in at.subheader]
    # the synthetic views still render (the leaderboard is now a subheader under the Proof-1 intro)
    assert "Leaderboard — ranked by Beat-the-Market" in subheaders
    # the historical views render
    assert "Historical Leaderboard" in headers
    assert "Historical Analyst Profile" in headers
    # (the historical call-level drill-down is now inside an expander; its table is counted below)

    # the historical analyst selector is populated from the sample (perma-bull + skilled present)
    hist_selectbox = next(sb for sb in at.selectbox if sb.label == "Choose an analyst (historical)")
    assert "Reed Calloway" in hist_selectbox.options
    assert "Ana Petrova" in hist_selectbox.options

    # synthetic + historical leaderboard & drill-down tables all rendered
    assert len(at.dataframe) >= 4
