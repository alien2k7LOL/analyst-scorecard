"""Analyst Scorecard — Streamlit demo app.

Run with:
    streamlit run app.py

Two tabs:
  • SYNTHETIC engine demo — the offline ground-truth dataset (leaderboard, per-analyst profile,
    call drill-down). No network or API key needed.
  • HISTORICAL back-test — grades real-style past calls (the shipped SAMPLE, or your own files)
    through the SAME look-ahead-safe engine: historical leaderboard, profiles vs the index over
    the historical span, call drill-down with the exact original-window prices, and full
    transparency on skipped / dropped calls.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from analyst_scorecard.backtest import SAMPLE_DATA_DIR, run_backtest
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.providers.live_web_price_provider import (
    LiveGradeError,
    grade_live_prediction,
)
from analyst_scorecard.schemas import Rating
from analyst_scorecard.verdicts import default_verdict_generator
from analyst_scorecard.viz import (
    build_dashboard,
    call_detail_dataframe,
    leaderboard_dataframe,
    plot_analyst_profile,
    plot_leaderboard,
    save_dashboard_pngs,
)

st.set_page_config(page_title="Analyst Scorecard", layout="wide")

st.title("📊 Analyst Scorecard")
st.caption(
    "Honest, fair, reproducible grading of analyst price targets. The headline metric is "
    "**beat-the-market**: would you have done better just buying the index?"
)

_LB_COLS = {
    "Beat-Market": st.column_config.NumberColumn(format="%.2f%%", help="Mean excess return vs index"),
    "Direction Hit-Rate": st.column_config.NumberColumn(format="%.0f%%"),
    "Accuracy": st.column_config.NumberColumn(format="%.3f"),
}

# Sidebar -----------------------------------------------------------------------------
seed = st.sidebar.number_input("Price-world seed (synthetic)", value=DEFAULT_CONFIG.seed, step=1)
config = DEFAULT_CONFIG.with_overrides(seed=int(seed))
if st.sidebar.button("Save synthetic charts as PNG"):
    paths = save_dashboard_pngs(config=config)
    st.sidebar.success("Saved:\n" + "\n".join(p.name for p in paths))
st.sidebar.divider()
data_dir = st.sidebar.text_input("Historical data folder", value=str(SAMPLE_DATA_DIR))


@st.cache_data(show_spinner=False)
def _load_synthetic(seed_value: int):
    cfg = DEFAULT_CONFIG.with_overrides(seed=seed_value)
    data = build_dashboard(cfg)
    gen = default_verdict_generator()
    verdicts = {aid: gen.verdict(s) for aid, s in data.scores_by_id.items()}
    return data, verdicts


@st.cache_data(show_spinner=False)
def _load_backtest(folder: str):
    result = run_backtest(folder)
    gen = default_verdict_generator()
    verdicts = {s.analyst_id: gen.verdict(s) for s in result.leaderboard.rows}
    return result, verdicts


def _profile_block(score, verdict_text: str):
    st.info(f"**Verdict:** {verdict_text}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Beat-the-Market", "—" if score.beat_market is None else f"{score.beat_market*100:+.1f}%")
    c2.metric("Direction Hit-Rate", f"{score.direction_hit_rate*100:.0f}%")
    c3.metric("Accuracy", "—" if score.mean_accuracy is None else f"{score.mean_accuracy:.3f}")
    st.pyplot(plot_analyst_profile(score))


tab_syn, tab_hist, tab_live = st.tabs(
    ["🧪 Synthetic engine demo", "📜 Historical back-test", "🛰️ Live Grader"]
)

# ===== Tab 1: synthetic ===============================================================
with tab_syn:
    data, verdicts = _load_synthetic(int(seed))

    st.header("Leaderboard — ranked by Beat-the-Market")
    st.dataframe(leaderboard_dataframe(data.leaderboard), use_container_width=True, hide_index=True, column_config=_LB_COLS)
    st.pyplot(plot_leaderboard(data.leaderboard))

    st.header("Analyst Profile")
    syn_names = {s.analyst_name: aid for aid, s in data.scores_by_id.items()}
    syn_choice = st.selectbox("Choose an analyst", [r.analyst_name for r in data.leaderboard.rows], key="syn_analyst")
    syn_aid = syn_names[syn_choice]
    _profile_block(data.scores_by_id[syn_aid], verdicts[syn_aid])

    st.header("Call-level drill-down — full traceability")
    st.caption("Every score traces to the exact call and the prices that resolved it.")
    st.dataframe(call_detail_dataframe(data.scores_by_id[syn_aid]), use_container_width=True, hide_index=True)

# ===== Tab 2: historical ==============================================================
with tab_hist:
    folder = Path(data_dir)
    if not (folder / "prices.csv").exists():
        st.error(
            f"No `prices.csv` found in `{data_dir}`. Point this at a folder with "
            "`prices.csv`, `calls.csv`, and `manifest.json` (see data/sample_historical/README.md)."
        )
    else:
        result, hist_verdicts = _load_backtest(str(folder))
        tag = "SAMPLE data — synthetic & fictional (replace with your own files)" if result.is_sample else "user-supplied data"
        st.caption(f"Source: **{tag}** · `{data_dir}`")
        if result.span_start:
            st.caption(f"Price span {result.span_start} → {result.span_end}")
        st.write(
            f"**{result.n_resolved}** calls resolved & scored · **{result.n_skipped}** skipped at "
            f"resolution · **{result.n_ingest_dropped}** dropped at ingest "
            f"(out of {result.n_ingested + result.n_ingest_dropped} raw rows)."
        )

        st.header("Historical Leaderboard")
        st.dataframe(leaderboard_dataframe(result.leaderboard), use_container_width=True, hide_index=True, column_config=_LB_COLS)
        st.pyplot(plot_leaderboard(result.leaderboard))

        st.header("Historical Analyst Profile")
        hist_names = {s.analyst_name: s.analyst_id for s in result.leaderboard.rows}
        hist_choice = st.selectbox("Choose an analyst (historical)", list(hist_names), key="hist_analyst")
        hist_aid = hist_names[hist_choice]
        _profile_block(result.analyst_scores[hist_aid], hist_verdicts[hist_aid])

        st.header("Historical Call-level Drill-down")
        st.caption("Each historical call and the exact original-window prices that resolved it.")
        st.dataframe(call_detail_dataframe(result.analyst_scores[hist_aid]), use_container_width=True, hide_index=True)

        with st.expander("Skipped & dropped calls (transparency — never silently scored)"):
            if result.skip_reason_counts:
                st.write("**Skipped at resolution:** " + ", ".join(f"{k} × {v}" for k, v in result.skip_reason_counts.items()))
            if result.ingest_reason_counts:
                st.write("**Dropped at ingest:** " + ", ".join(f"{k} × {v}" for k, v in result.ingest_reason_counts.items()))
            if result.skipped:
                st.dataframe(
                    [{"call_id": s.call.call_id, "ticker": s.call.ticker, "reason": s.reason, "detail": s.detail} for s in result.skipped],
                    use_container_width=True, hide_index=True,
                )
            if result.ingest_issues:
                st.dataframe(result.ingest_issues, use_container_width=True, hide_index=True)

# ===== Tab 3: live grader =============================================================
with tab_live:
    st.header("Live Grader — grade a prediction against today's market")
    st.caption(
        "Fetches **real** adjusted daily prices via `yfinance` and runs your prediction through the "
        "*same* look-ahead-safe funnel. Needs internet + the optional `yfinance` package "
        "(`.venv/bin/pip install -r requirements-live.txt`)."
    )
    st.warning(
        "A call whose horizon hasn't elapsed is graded **PROVISIONAL** — a mark-to-market read "
        "*so far*, not a final score. And because it uses live prices, a live grade is a "
        "point-in-time snapshot: **not reproducible** like the back-test."
    )

    c1, c2, c3 = st.columns(3)
    live_ticker = c1.text_input("Ticker", value="AAPL", key="live_ticker")
    live_rating = c2.selectbox("Rating", [r.value for r in Rating], index=0, key="live_rating")
    live_target = c3.number_input("Price target ($)", min_value=0.01, value=250.0, step=1.0, key="live_target")

    c4, c5, c6 = st.columns(3)
    live_call_date = c4.date_input(
        "Call date (must be in the past)", value=date(2025, 1, 2), max_value=date.today(), key="live_call_date"
    )
    live_benchmark = c5.text_input("Benchmark symbol", value="^GSPC", key="live_benchmark")
    horizon_mode = c6.radio("Horizon", ["Open-ended (so far)", "By deadline date"], key="live_hmode")

    live_deadline = None
    if horizon_mode == "By deadline date":
        live_deadline = st.date_input(
            "Original deadline (the call's resolution date)", value=date.today(), key="live_deadline"
        )

    if st.button("Grade Live Prediction", type="primary", key="live_go"):
        try:
            with st.spinner("Fetching live market data and grading…"):
                result = grade_live_prediction(
                    ticker=live_ticker,
                    rating=live_rating,
                    target_price=float(live_target),
                    call_date=live_call_date,
                    resolution_date=live_deadline,
                    benchmark_symbol=(live_benchmark or "^GSPC").strip(),
                    asof=date.today(),
                )
        except LiveGradeError as e:
            st.error(str(e))
        except Exception as e:  # network / library surprises -> never crash the page
            st.error(f"Couldn't grade that prediction: {e}")
        else:
            badge = st.warning if result.provisional else st.success
            through = "the deadline" if not result.provisional else "the latest available trading day"
            badge(
                f"**{result.status}** — graded from {result.call.call_date} through "
                f"{result.graded_through} ({through}). Live snapshot; not reproducible."
            )
            st.info(f"**Verdict:** {result.verdict}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Beat-the-Market", "—" if result.beat_market is None else f"{result.beat_market*100:+.1f}%")
            m2.metric("Direction gate", "PASS ✅" if result.call_score.direction_pass else "FAIL ❌")
            m3.metric("Accuracy", "—" if result.call_score.accuracy is None else f"{result.call_score.accuracy:.3f}")

            r = result.resolution
            st.caption("Exact prices used to resolve it (full traceability — nothing past the grading date):")
            st.dataframe(
                [{
                    "Ticker": result.call.ticker,
                    "Rating": result.call.rating.value,
                    "Benchmark": result.benchmark_symbol,
                    "Call date": r.call_date.isoformat(),
                    "Graded through": r.resolution_date.isoformat(),
                    "Call price": round(r.call_price, 2),
                    "Target": round(r.target_price, 2),
                    "Actual price": round(r.actual_price, 2),
                    "Stock return": f"{r.stock_return*100:+.1f}%",
                    "Benchmark return": f"{r.benchmark_return*100:+.1f}%",
                    "Beat": "—" if result.call_score.beat is None else f"{result.call_score.beat*100:+.1f}%",
                }],
                use_container_width=True, hide_index=True,
            )
            st.pyplot(plot_analyst_profile(result.analyst_score))
