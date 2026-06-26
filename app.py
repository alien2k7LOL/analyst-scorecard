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

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst_scorecard.backtest import SAMPLE_DATA_DIR, run_backtest
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.forecast.backtest import ForecastGenConfig, run_forecast_backtest
from analyst_scorecard.forecast.live import grade_forecast_live
from analyst_scorecard.forecast.prediction import PredictionKind
from analyst_scorecard.providers.live_web_price_provider import (
    LiveGradeError,
    grade_live_prediction,
)
from analyst_scorecard.schemas import Direction, Rating
from analyst_scorecard.verdicts import default_verdict_generator
from analyst_scorecard.viz import (
    build_dashboard,
    call_detail_dataframe,
    leaderboard_dataframe,
    plot_analyst_profile,
    plot_leaderboard,
    plot_reliability,
    save_dashboard_pngs,
)

st.set_page_config(page_title="Analyst Scorecard", layout="wide")

st.title("📊 Analyst Scorecard")
st.caption(
    "Honest, fair, reproducible grading of analyst price targets. The headline metric is "
    "**beat-the-market**: would you have done better just buying the index?"
)

with st.expander("🧭 New here? What this is & how to read it"):
    st.markdown(
        "- **What it does:** grades stock-call track records *and* your own forward predictions "
        "**honestly** — the engine never peeks at the future (look-ahead-safe and reproducible).\n"
        "- **Beat-the-Market** (the headline): mean excess return vs the index. **+40.5%** means "
        "following the calls beat just buying the index by 40.5 points; **negative** means it lagged.\n"
        "- **Direction Hit-Rate:** how often the up/down call was right. **Accuracy:** how close the "
        "price target landed (1.0 = bullseye).\n"
        "- **Where to start:** the **🔮 Forecast** tab grades *your own* prediction (type a ticker + a "
        "target). The first two tabs prove the method is fair on data whose outcomes are already known."
    )

_LB_COLS = {
    "Beat-Market": st.column_config.NumberColumn(
        format="%+.1f%%",
        help="The headline: mean excess return vs the index. Positive = following the calls beat the market.",
    ),
    "Direction Hit-Rate": st.column_config.NumberColumn(
        format="%.0f%%", help="How often the up/down call was correct."
    ),
    "Accuracy": st.column_config.NumberColumn(
        format="%.3f",
        help="Volatility-scaled closeness of target to reality (1.0 = bullseye); blank when no directional calls passed the gate.",
    ),
}


def _leaderboard_for_display(leaderboard):
    """Leaderboard dataframe with the two rate columns scaled to PERCENT (0.405 -> 40.5).

    Fixes the headline bug where column_config's '%%' format appended a sign to a FRACTION, so
    +40.5% rendered as '0.41%' and an 85% hit-rate as '1%'. Scaling here keeps numeric sorting.
    """
    df = leaderboard_dataframe(leaderboard).copy()
    for col in ("Beat-Market", "Direction Hit-Rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce") * 100.0
    return df

# Sidebar -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
seed = st.sidebar.number_input(
    "Price-world seed (synthetic)", value=DEFAULT_CONFIG.seed, step=1,
    help="Reseeds the synthetic price world on the demo tab. Doesn't affect the real-data tabs.",
)
config = DEFAULT_CONFIG.with_overrides(seed=int(seed))
if st.sidebar.button("Save synthetic charts as PNG"):
    paths = save_dashboard_pngs(config=config)
    st.sidebar.success("Saved:\n" + "\n".join(p.name for p in paths))
st.sidebar.divider()
with st.sidebar.expander("Advanced: use your own data"):
    st.caption("Most people can leave this as-is — it defaults to the bundled sample.")
    data_dir = st.text_input("Historical data folder", value=str(SAMPLE_DATA_DIR))


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
    c1.metric("Beat-the-Market", "—" if score.beat_market is None else f"{score.beat_market*100:+.1f}%",
              help="Mean excess return vs the index across this analyst's directional calls. Positive = added value over just buying the index.")
    c2.metric("Direction Hit-Rate", f"{score.direction_hit_rate*100:.0f}%",
              help="How often the up/down call was correct.")
    c3.metric("Accuracy", "—" if score.mean_accuracy is None else f"{score.mean_accuracy:.3f}",
              help="Volatility-scaled closeness of the target to reality (1.0 = bullseye).")
    st.pyplot(plot_analyst_profile(score, dark=True))


tab_syn, tab_hist, tab_live, tab_fcast = st.tabs(
    ["🧪 Synthetic engine demo", "📜 Historical back-test", "🛰️ Live Grader", "🔮 Forecast"]
)

# ===== Tab 1: synthetic ===============================================================
with tab_syn:
    data, verdicts = _load_synthetic(int(seed))

    st.header("Leaderboard — ranked by Beat-the-Market")
    st.caption("This is the offline demo: each row is a **fictional** analyst, and the engine grades their past calls. No real securities or people.")
    st.dataframe(_leaderboard_for_display(data.leaderboard), use_container_width=True, hide_index=True, column_config=_LB_COLS)
    st.pyplot(plot_leaderboard(data.leaderboard, dark=True))

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
    st.caption(
        "Grades real-style **past** calls whose outcomes are already known. The data source is set "
        "in the sidebar under **Advanced: use your own data** — it defaults to the bundled sample."
    )
    folder = Path(data_dir)
    if not (folder / "prices.csv").exists():
        st.error(
            f"No `prices.csv` in `{data_dir}`. Leave the sidebar on the bundled sample, or drop your "
            "own `prices.csv` / `calls.csv` / `manifest.json` into a folder and point the sidebar "
            "there (format guide: data/sample_historical/README.md)."
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
        st.dataframe(_leaderboard_for_display(result.leaderboard), use_container_width=True, hide_index=True, column_config=_LB_COLS)
        st.pyplot(plot_leaderboard(result.leaderboard, dark=True))

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
    live_ticker = c1.text_input("Ticker", value="AAPL", key="live_ticker", help="e.g. AAPL, MSFT, NVDA")
    live_rating = c2.selectbox("Rating", [r.value for r in Rating], index=0, key="live_rating")
    live_target = c3.number_input("Target price ($)", min_value=0.01, value=250.0, step=1.0, key="live_target")

    c4, c5, c6 = st.columns(3)
    live_call_date = c4.date_input(
        "Call date (must be in the past)", value=date(2025, 1, 2), max_value=date.today(), key="live_call_date"
    )
    live_benchmark = c5.text_input("Benchmark symbol", value="^GSPC", key="live_benchmark",
                                   help="Index to compare against — ^GSPC is the S&P 500.")
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
            st.pyplot(plot_analyst_profile(result.analyst_score, dark=True))

# ===== Tab 4: forecast (probability of a FUTURE prediction) ===========================
with tab_fcast:
    st.header("Forecast — probability a future prediction comes true")
    st.caption(
        "Estimate the probability a price **is at your target on the deadline** (terminal) or **reaches "
        "it any time before** (touch). The raw estimate comes from the stock's own drift + volatility "
        "(a closed-form GBM model); it's then **self-calibrated on that ticker's own history** so the "
        "number reflects how this stock has actually behaved. Needs internet + `yfinance`."
    )
    st.warning(
        "This is a **calibrated estimate, not a crystal ball** (a live, point-in-time snapshot — not "
        "reproducible). Judge it by **calibration** (does 70% mean 70%?) and **discrimination** (AUC) "
        "— **not “% accuracy.”** On our held-out sample backtest (**48k blind predictions**) the model "
        "stays well-calibrated (**ECE ≈ 0.01**) with **AUC ≈ 0.67** for the harder *terminal* (“at the "
        "price on the day”) calls and **≈ 0.85** for *touch* calls — both with news (0.5 = coin flip). "
        "The live path is **price-only**; the news lift is the research demo below, not applied here."
    )
    st.caption("Two ways to ask it: *“will it **be at** \\$300 (±3%) **on** my deadline?”* (terminal) or "
               "*“will it **touch** \\$300 **any time** before then?”* (touch) — each with a calibration "
               "curve showing how trustworthy the number is.")

    kind_label = st.radio(
        "What are you predicting?",
        ["🎯 It will **be at** this price ON the deadline",
         "📈 It will **touch** this price ANY TIME before the deadline"],
        key="f_kind", horizontal=True,
    )
    is_terminal = kind_label.startswith("🎯")

    c1, c2, c3 = st.columns(3)
    f_ticker = c1.text_input("Ticker", value="AAPL", key="f_ticker", help="e.g. AAPL, MSFT, NVDA")
    f_dir_label = c2.selectbox("Direction", ["UP — rises to target", "DOWN — falls to target"], key="f_dir",
                               help="Which way you expect it to move to reach the target. Orients the trend/news signals.")
    f_target = c3.number_input("Target price ($)", min_value=0.01, value=300.0, step=1.0, key="f_target")

    c4, c5, c6 = st.columns(3)
    f_deadline = c4.date_input("Deadline (a specific date)", value=date.today() + timedelta(days=180),
                               min_value=date.today() + timedelta(days=1), key="f_deadline",
                               help="The exact resolution date. Terminal mode grades the close ON this day.")
    f_bench = c5.text_input("Benchmark symbol", value="^GSPC", key="f_bench",
                            help="Index to compare against — ^GSPC is the S&P 500.")
    f_years = c6.slider("Years of history to self-calibrate on", 3, 10, 6, key="f_years")

    if is_terminal:
        f_band_pct = st.slider("How close counts? Tolerance band around the target (±%)", 1, 15, 3,
                               key="f_band") / 100.0
        lo, hi = f_target * (1 - f_band_pct), f_target * (1 + f_band_pct)
        st.caption(f"A **hit** = the closing price on **{f_deadline}** lands within **±{int(f_band_pct*100)}%** of "
                   f"\\${f_target:,.2f} (i.e. \\${lo:,.2f}–\\${hi:,.2f}). Landing at the target on the day is a "
                   "*harder* call than ever touching it — so expect lower, more honest probabilities.")
    else:
        f_band_pct = None

    if st.button("Estimate probability", type="primary", key="f_go"):
        direction = Direction.UP if f_dir_label.startswith("UP") else Direction.DOWN
        kind = PredictionKind.TERMINAL if is_terminal else PredictionKind.TOUCH
        try:
            with st.spinner("Fetching history, self-calibrating, and grading…"):
                g = grade_forecast_live(
                    ticker=f_ticker, target_price=float(f_target), deadline=f_deadline,
                    direction=direction, kind=kind, band_pct=f_band_pct,
                    benchmark_symbol=(f_bench or "^GSPC").strip(),
                    years_history=int(f_years), as_of=date.today(),
                )
        except LiveGradeError as e:
            st.error(str(e))
        except Exception as e:  # network / library surprises -> never crash the page
            st.error(f"Couldn't grade that prediction: {e}")
        else:
            if g.kind == PredictionKind.TERMINAL:
                b = g.band_pct or 0.0
                headline = (f"P({g.ticker} closes within ±{int(b*100)}% of ${g.target_price:,.2f} "
                            f"on {g.deadline}) ≈ {g.probability*100:.0f}%")
            else:
                verb = "rise to" if g.direction == Direction.UP else "fall to"
                headline = f"P({g.ticker} will {verb} ${g.target_price:,.2f} by {g.deadline}) ≈ {g.probability*100:.0f}%"
            st.subheader(headline)
            m1, m2, m3 = st.columns(3)
            m1.metric("Calibrated probability", f"{g.probability*100:.0f}%",
                      help="The self-calibrated estimate — corrected by how this stock has actually behaved.")
            m2.metric("Raw model (uncalibrated)", f"{g.raw_probability*100:.0f}%",
                      help="Straight from the GBM model, before history-based correction.")
            m3.metric("Now → deadline", f"{g.n_days} trading days")
            if g.calibrated:
                rec = g.self_cal_metrics.get("recalibrated", {})
                best = (g.self_cal_metrics.get("full+") or g.self_cal_metrics.get("full")
                        or g.self_cal_metrics.get("+regime") or g.self_cal_metrics.get("+momentum") or rec)
                ece, auc, n_test = rec.get("ece"), (best or {}).get("auc"), rec.get("n")
                cred = f"When this model says 60%, it has historically hit ~60% on {g.ticker}"
                if ece is not None and auc is not None:
                    cred += f" (calibration error ≈ {ece:.2f}, AUC ≈ {auc:.2f} — 0.5 is a coin flip)"
                st.success(
                    f"Self-calibrated on {g.ticker}'s own history {g.history_start} → {g.history_end}. " + cred + "."
                )
                if n_test is not None and n_test < 60:
                    st.caption(f"⚠️ Thin sample ({n_test} held-out windows on one stock) — treat the "
                               "curve and the probability as **indicative**, not precise.")
                if g.reliability:
                    st.pyplot(plot_reliability(g.reliability, dark=True))
            else:
                st.info("Not enough history to self-calibrate — showing the raw model probability (uncalibrated).")
            st.caption(
                f"Start price ${g.s0:,.2f} · benchmark {g.benchmark_symbol} · "
                f"as of {g.as_of}. Look-ahead-safe: only data up to today informs the estimate."
            )

            # Persist this prediction for the session so it isn't a throwaway calculation.
            st.session_state.setdefault("forecasts", []).append({
                "saved": str(g.as_of),
                "ticker": g.ticker,
                "type": "at-the-date" if g.kind == PredictionKind.TERMINAL else "touch-before",
                "direction": g.direction.value,
                "target": round(g.target_price, 2),
                "band_±%": (int(g.band_pct * 100) if g.band_pct else None),
                "deadline": str(g.deadline),
                "probability_%": round(g.probability * 100),
                "calibrated": g.calibrated,
            })

    # ---- My predictions (persisted for this session; downloadable) ----
    saved = st.session_state.get("forecasts", [])
    if saved:
        st.divider()
        st.subheader("📌 My predictions (this session)")
        st.caption("Saved as you estimate. Download to keep them — a forecast you can't revisit is just a calculator.")
        st.dataframe(pd.DataFrame(saved), use_container_width=True, hide_index=True)
        cdl, ccl = st.columns(2)
        cdl.download_button("⬇️ Download as JSON", data=json.dumps(saved, indent=2),
                            file_name="my_predictions.json", mime="application/json", key="fc_dl")
        if ccl.button("Clear", key="fc_clear"):
            st.session_state["forecasts"] = []
            st.rerun()

    # ---- evidence: the offline sample shows point-in-time NEWS adds value ----
    with st.expander("Evidence — does point-in-time news actually help? (offline sample backtest)"):
        st.caption(
            "The live path is price-only. On the shipped SAMPLE (synthetic, with a timestamped news "
            "feed), the calibration backtest measures whether point-in-time news improves held-out "
            "accuracy — news is only credited if it beats the price-only model on data it never saw."
        )

        @st.cache_data(show_spinner=False)
        def _sample_forecast():
            r = run_forecast_backtest("data/sample_historical", with_news=True,
                                      gen=ForecastGenConfig(stride_days=21))
            order = ["base_rate", "raw", "recalibrated", "+momentum", "+regime", "+news", "full", "full+"]
            table = [{"model": k, "Brier": round(v["brier"], 4), "LogLoss": round(v["log_loss"], 4),
                      "ECE": round(v["ece"], 4), "AUC": round(v["auc"], 3)}
                     for k in order if k in r.metrics
                     for v in [r.metrics[k]]]
            return table, r.news_helps, r.reliability

        if st.button("Run the sample calibration backtest", key="f_sample"):
            with st.spinner("Backtesting the sample…"):
                table, news_helps, reliability = _sample_forecast()
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.write(f"**Point-in-time news adds held-out value:** {'✅ yes' if news_helps else '❌ no'} "
                     "(lower Brier / LogLoss / ECE is better).")
            st.pyplot(plot_reliability(reliability, dark=True))
