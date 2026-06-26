"""Phase 7 — visualization: dashboard data, charts, PNG export, traceability."""

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.viz import (
    build_dashboard,
    call_detail_dataframe,
    leaderboard_dataframe,
    plot_analyst_profile,
    plot_leaderboard,
    save_dashboard_pngs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard():
    return build_dashboard(DEFAULT_CONFIG)


def test_leaderboard_dataframe_is_ranked_by_beat_market(dashboard):
    df = leaderboard_dataframe(dashboard.leaderboard)
    assert list(df["Rank"]) == sorted(df["Rank"])
    # ranked desc by beat-market -> the skilled picker is at the top
    assert df.iloc[0]["Analyst"] == "Vega Capital"
    assert {"Beat-Market", "Direction Hit-Rate", "Accuracy"}.issubset(df.columns)


def test_call_detail_shows_exact_resolving_prices(dashboard):
    """The drill-down prices must be the ACTUAL prices the resolver used (traceability)."""
    score = dashboard.scores_by_id["vega"]
    df = call_detail_dataframe(score)
    assert len(df) == score.n_calls
    provider = dashboard.provider
    for _, row in df.iterrows():
        call_price = provider.price_on(row["Ticker"], row["Call Date"])
        actual_price = provider.price_on(row["Ticker"], row["Resolution Date"])
        assert row["Call Price"] == pytest.approx(round(call_price, 2))
        assert row["Actual Price"] == pytest.approx(round(actual_price, 2))


def test_profile_chart_renders_for_skilled_and_rider(dashboard):
    for aid in ("vega", "momentum", "hubris"):
        fig = plot_analyst_profile(dashboard.scores_by_id[aid])
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)


def test_leaderboard_chart_renders(dashboard):
    fig = plot_leaderboard(dashboard.leaderboard)
    assert fig is not None
    plt.close(fig)


def test_save_dashboard_pngs_writes_nonempty_files(tmp_path):
    paths = save_dashboard_pngs(outdir=tmp_path, config=DEFAULT_CONFIG)
    assert len(paths) >= 2  # leaderboard + at least one profile
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    assert any("leaderboard" in p.name for p in paths)
    assert any("profile_" in p.name for p in paths)


def test_streamlit_app_compiles():
    """app.py must be valid Python and importable structure (syntax check)."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(REPO_ROOT / "app.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_streamlit_app_renders_all_three_views():
    """Run the app in-process (AppTest) and confirm it renders without exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120).run()
    assert list(at.exception) == [], at.exception
    subheaders = [h.value for h in at.subheader]
    assert "Leaderboard — ranked by Beat-the-Market" in subheaders
    assert "Analyst Profile" in subheaders
    # the call-level drill-down is now inside an expander; its table still renders below
    assert len(at.dataframe) >= 2
    # analyst selector is populated, ranked by beat-market (skilled picker first).
    # Find it by content (robust to tab order) rather than a positional index.
    syn_box = next(sb for sb in at.selectbox if "Vega Capital" in sb.options)
    assert syn_box.options[0] == "Vega Capital"
