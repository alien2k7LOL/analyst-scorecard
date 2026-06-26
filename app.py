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
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst_scorecard.backtest import SAMPLE_DATA_DIR, run_backtest
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.forecast.backtest import ForecastGenConfig, run_forecast_backtest
from analyst_scorecard.forecast.explanation import build_math, formula_text, news_lean, recent_headlines
from analyst_scorecard.forecast.interval import BarInterval
from analyst_scorecard.forecast.live import grade_forecast_live
from analyst_scorecard.forecast.prediction import PredictionKind
from analyst_scorecard.intel.extract import (
    ExtractedRecommendation,
    UrlFetchError,
    extract_from_url,
    extract_recommendation,
)
from analyst_scorecard.intel.report import build_report
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
    plot_forecast_proof,
    plot_leaderboard,
    plot_recommendation_chart,
    plot_reliability,
    save_dashboard_pngs,
)

st.set_page_config(page_title="Analyst Scorecard", layout="wide")

st.title("📊 Analyst Scorecard")
st.caption(
    "**Is a price prediction actually likely — and should you trust it?**  Enter a target — yours or an "
    "analyst's — and get a **calibrated probability** with the math, price-cone chart, and news behind it. "
    "Calibrated on 48k+ backtested predictions (see the *Proof* tabs)."
)

with st.expander("🧭 New here? Start here (30-second orientation)", expanded=False):
    st.markdown(
        "**One question: how likely is a price prediction, and can you trust the number?** Two ways in, "
        "plus the evidence:\n\n"
        "- 🎯 **Forecast a prediction** — *your* call. Enter ticker + target + deadline → a probability, a "
        "plain-English verdict, and the proof (math, price cone, supporting/contradicting news). **Start here.**\n"
        "- 🛰️ **Analyze a call** — *an analyst's* recommendation. Paste the text or a URL → it pulls out the "
        "call and grades it against the live market, with a forward probability.\n"
        "- 🔬 **Proof tabs** — *why trust the numbers.* Evidence the engine catches real skill and stays "
        "calibrated (when it says 30%, it happens ~30%). Optional — skip unless you want to kick the tyres.\n\n"
        "Everything is look-ahead-safe and reproducible — the engine never peeks at the future."
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


def _render_forecast_result(g):
    """Render a live forecast grade: headline, credibility, and the math/graph/news PROOF."""
    unit = g.interval.horizon_word
    if g.kind == PredictionKind.TERMINAL:
        b = g.band_pct or 0.0
        headline = (f"P({g.ticker} is within ±{b*100:g}% of \\${g.target_price:,.2f} "
                    f"on {g.deadline_label}) ≈ {g.probability*100:.0f}%")
    else:
        verb = "rise to" if g.direction == Direction.UP else "fall to"
        headline = f"P({g.ticker} will {verb} \\${g.target_price:,.2f} by {g.deadline_label}) ≈ {g.probability*100:.0f}%"
    st.subheader(headline)

    m1, m2, m3 = st.columns(3)
    m1.metric("Calibrated probability", f"{g.probability*100:.0f}%",
              help="The self-calibrated estimate — corrected by how this stock has actually behaved.")
    m2.metric("Raw model (uncalibrated)", f"{g.raw_probability*100:.0f}%",
              help="Straight from the closed-form GBM model, before history-based correction.")
    m3.metric("Now → deadline", f"{g.n_days} {unit}")

    if g.calibrated:
        rec = g.self_cal_metrics.get("recalibrated", {})
        best = (g.self_cal_metrics.get("full+") or g.self_cal_metrics.get("full")
                or g.self_cal_metrics.get("+regime") or g.self_cal_metrics.get("+momentum") or rec)
        ece, auc, n_test = rec.get("ece"), (best or {}).get("auc"), rec.get("n")
        cred = f"When this model says 60%, it has historically hit ~60% on {g.ticker}"
        if ece is not None and auc is not None:
            cred += f" (calibration error ≈ {ece:.2f}, AUC ≈ {auc:.2f} — 0.5 is a coin flip)"
        st.success(f"Self-calibrated on {g.ticker}'s own {g.interval.label} history "
                   f"{g.history_start} → {g.history_end}. " + cred + ".")
        if n_test is not None and n_test < 60:
            st.caption(f"⚠️ Thin sample ({n_test} held-out windows on one stock) — treat the curve and "
                       "the probability as **indicative**, not precise.")
        if g.interval == BarInterval.MIN30:
            st.caption("⚠️ Intraday calibration leans on only ~60 days of 30-min bars — thinner and "
                       "noisier than the daily model. Treat 30-min probabilities as directional.")
        if g.reliability:
            st.pyplot(plot_reliability(g.reliability, dark=True))
    else:
        st.info("Not enough history to self-calibrate — showing the raw model probability (uncalibrated).")
    st.caption(f"Start price \\${g.s0:,.2f} · benchmark {g.benchmark_symbol} · as of {g.as_of} · "
               f"{g.interval.label} resolution. Look-ahead-safe: only data up to now informs the estimate.")

    # ---------------- THE PROOF: maths + picture + news ----------------
    st.divider()
    st.subheader("🔍 Why this number? — the proof")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**The maths** — the ingredients behind the number")
        st.dataframe(
            pd.DataFrame([{"factor": f.label, "value": f.value, "what it means": f.meaning}
                          for f in build_math(g)]),
            hide_index=True, use_container_width=True,
        )
        st.caption("Closed form behind the raw number:")
        st.code(formula_text(g), language=None)
    with right:
        verb = "rise" if g.direction == Direction.UP else "fall"
        st.markdown(f"**News check** — does recent coverage back this *{verb}* call?")
        try:
            heads = recent_headlines(g.ticker)
        except Exception:
            heads = []
        if heads:
            lean = news_lean(heads, g.direction)
            st.caption(f"✅ {lean['supports']} support · ❌ {lean['contradicts']} contradict · "
                       f"➖ {lean['neutral']} neutral  (recent headlines vs your call)")
            for h, lab in zip(heads[:5], lean["labels"]):
                tag = {"supports": "✅", "contradicts": "❌", "neutral": "➖"}[lab]
                line = f"{tag} **{h.title}**"
                if h.publisher:
                    line += f"  ·  {h.publisher}"
                st.markdown(line)
            st.caption("Shown as supporting/contradicting evidence — it explains the call, and isn't fed "
                       "into the probability (that would risk look-ahead in the self-calibration).")
        else:
            st.caption("No recent headlines available (offline, or none returned for this ticker).")
    st.markdown("**The picture** — the price cone and where the probability comes from")
    st.pyplot(plot_forecast_proof(g, dark=True))

    # Persist this prediction for the session so it isn't a throwaway calculation.
    st.session_state.setdefault("forecasts", []).append({
        "saved": str(g.as_of),
        "ticker": g.ticker,
        "resolution": g.interval.value,
        "type": "at-the-date" if g.kind == PredictionKind.TERMINAL else "touch-before",
        "direction": g.direction.value,
        "target": round(g.target_price, 2),
        "band_±%": (round(g.band_pct * 100, 2) if g.band_pct else None),
        "deadline": g.deadline_label,
        "probability_%": round(g.probability * 100),
        "calibrated": g.calibrated,
    })


def _intel_summary_block(rec):
    """Section 1 — what we extracted, shown for confirmation (override via Manual entry)."""
    st.markdown("#### 1 · Recommendation summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Ticker", rec.ticker or "—")
    s2.metric("Rating", rec.rating or "—")
    s3.metric("Target", f"${rec.target_price:,.2f}" if rec.target_price else "—")
    s4.metric("Extracted by", rec.source.upper())
    meta = []
    if rec.analyst:
        meta.append(f"**Analyst:** {rec.analyst}")
    if rec.firm:
        meta.append(f"**Firm:** {rec.firm}")
    meta.append(f"**Call date:** {rec.publication_date or 'not stated (assumed)'}")
    st.caption(" · ".join(meta))
    if rec.notes:
        st.caption("⚠ " + "; ".join(rec.notes))


def _render_intel_sections(rep):
    """Sections 2–6 of the analyst report from a pre-built AnalystReport."""
    rec = rep.rec

    st.markdown("#### 2 · The takeaway")
    st.info(rep.summary.replace("$", "\\$"))  # escape $ so Streamlit doesn't read it as LaTeX

    st.markdown("#### 3 · Live scorecard")
    if rep.live is None:
        st.warning(f"Live market data unavailable — {rep.live_error}")
    else:
        lv = rep.live
        badge = st.warning if lv.provisional else st.success
        badge(f"**{lv.status}** — graded {rep.call_date} → {lv.graded_through}"
              + (" · call date assumed" if rep.call_date_assumed else ""))
        a, b, c, d = st.columns(4)
        a.metric("Current price", f"${lv.current_price:,.2f}", f"{lv.return_since_pub*100:+.1f}% since call")
        b.metric("Alpha vs benchmark", "—" if lv.alpha is None else f"{lv.alpha*100:+.1f}%",
                 help="Excess return of following the call vs just holding the index.")
        c.metric("Distance to target", f"{lv.distance_to_target*100:+.1f}%",
                 help="How far today's price is from the target (positive = target above price).")
        d.metric("Days held", f"{lv.days_since}")
        e, f2, g2 = st.columns(3)
        e.metric("Stock return", f"{lv.return_since_pub*100:+.1f}%")
        f2.metric("Benchmark return", f"{lv.benchmark_return*100:+.1f}%")
        g2.metric("Direction gate", "PASS ✅" if lv.direction_pass else "FAIL ❌")
    if rep.forward is not None:
        st.metric(f"Model odds of reaching \\${rec.target_price:,.0f} within ~12 months",
                  f"{rep.forward.probability*100:.0f}%",
                  help="From the calibrated forecast engine (touch-by-deadline), self-calibrated on this stock.")
    elif rep.forward_error:
        st.caption(f"Forward outlook: {rep.forward_error}")

    st.markdown("#### 4 · Historical analyst performance")
    h = rep.historical
    if h is None:
        st.caption("No analyst name was extracted — paste a note that names the analyst to see a track record.")
    elif not h.found:
        st.caption(f"No track record on file for **{h.analyst}**. The bundled dataset is the synthetic "
                   "sample; point the app at a real recommendations dataset to populate this.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Direction win-rate", f"{h.win_rate*100:.0f}%" if h.win_rate is not None else "—")
        b.metric("Avg alpha", f"{h.avg_alpha*100:+.1f}%" if h.avg_alpha is not None else "—")
        c.metric("Graded calls", f"{h.n_recs}")
        on_stock = f"{h.on_this_stock_n}" + (f" · {h.on_this_stock_avg_alpha*100:+.1f}%"
                                             if h.on_this_stock_avg_alpha is not None else "")
        d.metric(f"On {rec.ticker}", on_stock)

    st.markdown("#### 5 · Price since the call")
    st.pyplot(plot_recommendation_chart(rep, dark=True))

    st.markdown("#### 6 · Similar historical calls")
    if rep.similar:
        st.caption("Resolved calls from the loaded dataset on the same ticker or direction — and how they fared:")
        st.dataframe([{
            "Ticker": s.ticker, "Rating": s.rating, "Call": s.call_date.isoformat(),
            "Resolved": s.resolution_date.isoformat(),
            "Stock %": round(s.stock_return * 100, 1), "Bench %": round(s.benchmark_return * 100, 1),
            "Beat %": None if s.beat is None else round(s.beat * 100, 1),
        } for s in rep.similar], hide_index=True, use_container_width=True)
    else:
        st.caption("No comparable calls in the loaded dataset (its tickers are synthetic sample names).")


# Trader-first order: the decision tools lead; the credibility proofs come last.
tab_fcast, tab_live, tab_syn, tab_hist = st.tabs(
    ["🎯 Forecast a prediction", "🛰️ Analyze an analyst's call",
     "🔬 Proof · catches skill", "📜 Proof · calibrated on history"]
)

# ===== Tab 1: synthetic ===============================================================
with tab_syn:
    st.info(
        "**Proof #1 — can the engine spot skill when we *know* the truth?**  \n"
        "Real markets never tell you who is genuinely skilled, so here we **invent** a market where we "
        "do: some fictional analysts are built to have real skill, others are hype. They run through the "
        "**same engine** that grades real calls. If skill rises to the top, the engine is trustworthy.  \n"
        "👉 *What to look for:* green (beat-the-market) bars belong to the skilled analysts — then "
        "**re-roll the world** below and watch the ranking hold."
    )
    data, verdicts = _load_synthetic(int(seed))

    st.subheader("Leaderboard — ranked by Beat-the-Market")
    st.caption("Each row is a **fictional** analyst; the engine grades their past calls. No real securities or people.")
    st.dataframe(_leaderboard_for_display(data.leaderboard), use_container_width=True, hide_index=True, column_config=_LB_COLS)
    st.pyplot(plot_leaderboard(data.leaderboard, dark=True))

    with st.expander("🎲 Robustness check — re-roll the world: does the ranking hold? (skill ≠ luck)"):
        st.caption("Each re-roll is a brand-new random market (a different price seed). If the same analyst "
                   "keeps winning, the engine is reading **persistent skill**, not the luck of one set of prices.")
        if st.button("Re-roll 6 different worlds", key="reroll"):
            from collections import Counter
            tops = []
            with st.spinner("Generating 6 fresh markets and re-grading…"):
                for s in range(int(seed), int(seed) + 6):
                    d, _ = _load_synthetic(s)
                    rows = [r for r in d.leaderboard.rows if r.beat_market is not None]
                    if rows:
                        tops.append(max(rows, key=lambda r: r.beat_market).analyst_name)
            if tops:
                winner, cnt = Counter(tops).most_common(1)[0]
                st.metric("Most frequent #1 across 6 random worlds", winner, f"{cnt}/6 worlds")
                st.caption("One analyst dominating the top spot → the engine detects real skill. A different "
                           "winner each time would mean it's just reading noise.")

    st.subheader("Analyst Profile")
    syn_names = {s.analyst_name: aid for aid, s in data.scores_by_id.items()}
    syn_choice = st.selectbox("Choose an analyst", [r.analyst_name for r in data.leaderboard.rows], key="syn_analyst")
    syn_aid = syn_names[syn_choice]
    _profile_block(data.scores_by_id[syn_aid], verdicts[syn_aid])

    with st.expander("🔬 Call-level drill-down — full traceability (every score traces to the exact prices)"):
        st.dataframe(call_detail_dataframe(data.scores_by_id[syn_aid]), use_container_width=True, hide_index=True)

# ===== Tab 2: historical ==============================================================
with tab_hist:
    st.info(
        "**Proof #2 — is it honest on *real-shaped* history?**  \n"
        "This grades real-style **past** calls whose outcomes already happened, under one ironclad rule: "
        "**no look-ahead** — it never sees a price past a call's resolution date. Calls it can't fairly "
        "resolve are **skipped and shown**, never quietly scored.  \n"
        "👉 *What to look for:* the beat-the-market spread, and the full transparency of skipped / dropped "
        "calls at the bottom. (Same engine as Proof #1 — now on real-shaped data.)"
    )
    st.caption("Data source is set in the sidebar under **Advanced: use your own data** — defaults to the bundled sample.")
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

        with st.expander("🔬 Historical call-level drill-down — each call and the exact prices that resolved it"):
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
    st.header("🛰️ Is this analyst's call worth following?")
    st.success(
        "**You paste:** an analyst recommendation (a headline, tweet, research note, or article URL).  →  "
        "**You get:** the extracted call + how it's done against the real market since, the firm's track "
        "record, and a forward probability of the target being hit."
    )
    st.caption(
        "We pull out the call (analyst, ticker, rating, target, date) and grade it against **real** prices "
        "with the same look-ahead-safe engine. Live prices need internet + `yfinance`."
    )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.caption("ℹ️ Extraction is using the **offline heuristic parser** (no `ANTHROPIC_API_KEY` set) — "
                   "set the key and install `anthropic` for LLM-grade extraction.")

    paste = st.text_area(
        "Paste an analyst recommendation…", height=130, key="intel_paste",
        placeholder="e.g. “Wedbush's Dan Ives reiterates Buy on Apple (AAPL) and raises his price target to $300.”",
    )
    cu1, cu2 = st.columns([3, 1])
    url = cu1.text_input("…or paste an article URL", key="intel_url", placeholder="https://…")
    intel_bench = cu2.text_input("Benchmark", value="^GSPC", key="intel_bench")
    analyze = st.button("Analyze Recommendation", type="primary", key="intel_go")

    # Historical context comes from the loaded back-test (cached); used to match the analyst/ticker.
    try:
        intel_hist, _ = _load_backtest(str(SAMPLE_DATA_DIR))
    except Exception:
        intel_hist = None

    triggered = None
    if analyze:
        try:
            if url.strip():
                with st.spinner("Fetching the article…"):
                    text = extract_from_url(url.strip())
            else:
                text = paste
            if not (text and text.strip()):
                st.error("Paste some recommendation text, or a URL, first.")
            else:
                with st.spinner("Extracting the recommendation…"):
                    triggered = extract_recommendation(text)
        except UrlFetchError as e:
            st.error(str(e))
        except Exception as e:  # never crash the page
            st.error(f"Couldn't analyze that: {e}")

    with st.expander("⚙️ Manual entry (advanced) — type the fields directly, or fix a misread one"):
        m1, m2, m3 = st.columns(3)
        mt = m1.text_input("Ticker", key="m_ticker")
        mr = m2.selectbox("Rating", [""] + [r.value for r in Rating], key="m_rating")
        mtg = m3.number_input("Target price ($)", min_value=0.0, value=0.0, step=1.0, key="m_target")
        m4, m5, m6 = st.columns(3)
        man = m4.text_input("Analyst (optional)", key="m_analyst")
        mfirm = m5.text_input("Firm (optional)", key="m_firm")
        mdate = m6.date_input("Call date", value=date.today() - timedelta(days=120),
                              max_value=date.today(), key="m_date")
        if st.button("Grade manual entry", key="m_go"):
            triggered = ExtractedRecommendation(
                ticker=(mt.strip().upper() or None), rating=(mr or None),
                target_price=(float(mtg) or None), analyst=(man.strip() or None),
                firm=(mfirm.strip() or None), publication_date=mdate, source="manual",
            )

    # On a trigger: extract + (if gradeable) build the report; persist both so the view is sticky.
    if triggered is not None:
        st.session_state["intel_rec"] = triggered
        if triggered.is_gradeable:
            with st.spinner("Grading against the live market and forecasting…"):
                try:
                    st.session_state["intel_report"] = build_report(
                        triggered, benchmark_symbol=(intel_bench or "^GSPC").strip(),
                        historical_result=intel_hist,
                    )
                except LiveGradeError as e:
                    st.session_state["intel_report"] = None
                    st.error(str(e))
                except Exception as e:
                    st.session_state["intel_report"] = None
                    st.error(f"Couldn't build the report: {e}")
        else:
            st.session_state["intel_report"] = None

    rec = st.session_state.get("intel_rec")
    if rec is not None:
        st.divider()
        _intel_summary_block(rec)
        if not rec.is_gradeable:
            st.error("Missing **" + ", ".join(rec.missing_fields()) + "** — add them via Manual entry above.")
        else:
            rep = st.session_state.get("intel_report")
            if rep is not None:
                _render_intel_sections(rep)

# ===== Tab 4: forecast (probability of a FUTURE prediction) ===========================
with tab_fcast:
    st.header("🎯 Will this price target happen?")
    st.success(
        "**You enter:** a ticker, a target price, and a deadline.  →  **You get:** a calibrated probability, "
        "a plain-English verdict, and the proof behind it (the math, a price-cone chart, and recent news "
        "for/against). *This is the main tool — start here.*"
    )
    st.caption(
        "Under the hood: the probability comes from the stock's own volatility (a closed-form model), then "
        "is **self-calibrated on that ticker's own history** so it reflects how the stock has actually "
        "behaved. Needs internet + `yfinance`."
    )
    st.warning(
        "This is a **calibrated estimate, not a crystal ball** (a live, point-in-time snapshot — not "
        "reproducible). Judge it by **calibration** (does 70% mean 70%?) and **discrimination** (AUC) "
        "— **not “% accuracy.”** On held-out sample backtests the model stays well-calibrated "
        "(**ECE ≈ 0.01**): **daily** terminal **AUC ≈ 0.67** (48k blind predictions); **30-min intraday "
        "AUC ≈ 0.60** (thinner ~60-day history — the default, but weaker). 0.5 = a coin flip. "
        "The live path is **price-only**; news below is shown as context."
    )
    st.caption("Pick a **resolution** and a **question**, then get a calibrated probability with a full "
               "proof (the maths, a picture, and recent news).")

    ci1, ci2 = st.columns(2)
    interval_label = ci1.radio(
        "Resolution",
        ["⏱️ 30-min bars (intraday — default)", "📅 Daily (longer horizon)"],
        key="f_interval",
        help="30-min: near-term, intraday deadlines, calibrated on ~60 days of bars (thinner). "
             "Daily: weeks–months out, calibrated on years of history (stronger).",
    )
    is_intraday = interval_label.startswith("⏱️")
    interval = BarInterval.MIN30 if is_intraday else BarInterval.DAILY
    kind_label = ci2.radio(
        "What are you predicting?",
        ["🎯 It will **be at** this price ON the deadline",
         "📈 It will **touch** this price ANY TIME before the deadline"],
        key="f_kind",
    )
    is_terminal = kind_label.startswith("🎯")

    c1, c2, c3 = st.columns(3)
    f_ticker = c1.text_input("Ticker", value="AAPL", key="f_ticker", help="e.g. AAPL, MSFT, NVDA")
    f_dir_label = c2.selectbox("Direction", ["UP — rises to target", "DOWN — falls to target"], key="f_dir",
                               help="Which way you expect it to move to reach the target. Orients the trend signals.")
    f_target = c3.number_input("Target price ($)", min_value=0.01, value=300.0, step=1.0, key="f_target")

    if is_intraday:
        c4, c5, c6 = st.columns(3)
        d_date = c4.date_input("Deadline date", value=date.today() + timedelta(days=1),
                               min_value=date.today(), key="f_idate")
        d_time = c5.time_input("Deadline time (market-local)", value=time(12, 0), step=1800, key="f_itime",
                               help="Resolved on the 30-min bar at/just before this time.")
        f_bench = c6.text_input("Benchmark", value="^GSPC", key="f_bench")
        deadline = datetime.combine(d_date, d_time)
        years = 6
    else:
        c4, c5, c6 = st.columns(3)
        deadline = c4.date_input("Deadline (a specific date)", value=date.today() + timedelta(days=180),
                                 min_value=date.today() + timedelta(days=1), key="f_deadline",
                                 help="The exact resolution date. Terminal mode grades the close ON this day.")
        f_bench = c5.text_input("Benchmark", value="^GSPC", key="f_bench",
                                help="Index to compare against — ^GSPC is the S&P 500.")
        years = c6.slider("Years of history to self-calibrate on", 3, 10, 6, key="f_years")

    if is_terminal:
        if is_intraday:
            f_band_pct = st.slider("How close counts? Tolerance band (±%)", 0.1, 3.0, 0.6, 0.1, key="f_band_i") / 100.0
        else:
            f_band_pct = float(st.slider("How close counts? Tolerance band (±%)", 1, 15, 3, key="f_band")) / 100.0
        lo, hi = f_target * (1 - f_band_pct), f_target * (1 + f_band_pct)
        st.caption(f"A **hit** = the price at the deadline lands within **±{f_band_pct*100:g}%** of "
                   f"\\${f_target:,.2f} (i.e. \\${lo:,.2f}–\\${hi:,.2f}). Landing AT the target is a *harder* "
                   "call than ever touching it — so expect lower, more honest probabilities.")
    else:
        f_band_pct = None

    if st.button("Estimate probability", type="primary", key="f_go"):
        direction = Direction.UP if f_dir_label.startswith("UP") else Direction.DOWN
        kind = PredictionKind.TERMINAL if is_terminal else PredictionKind.TOUCH
        try:
            with st.spinner("Fetching history, self-calibrating, and grading…"):
                g = grade_forecast_live(
                    ticker=f_ticker, target_price=float(f_target), deadline=deadline,
                    direction=direction, kind=kind, band_pct=f_band_pct, interval=interval,
                    benchmark_symbol=(f_bench or "^GSPC").strip(),
                    years_history=int(years), as_of=date.today(),
                )
        except LiveGradeError as e:
            st.error(str(e))
        except Exception as e:  # network / library surprises -> never crash the page
            st.error(f"Couldn't grade that prediction: {e}")
        else:
            _render_forecast_result(g)

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
