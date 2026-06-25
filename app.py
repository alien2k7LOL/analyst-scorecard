"""Analyst Scorecard — Streamlit demo app.

Run with:
    streamlit run app.py

Three views (all from the offline synthetic data — no network or API key needed):
  1. LEADERBOARD ranked by beat-the-market (the headline), with direction hit-rate & accuracy.
  2. Per-analyst PROFILE chart: following-the-call return vs the index, so a skilled analyst
     sits above the line and a hype machine below it; plus a plain-English verdict.
  3. Call-level DRILL-DOWN: the original call and the exact prices that resolved it.
"""

from __future__ import annotations

import streamlit as st

from analyst_scorecard.config import DEFAULT_CONFIG
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

# Sidebar — reproducible price world
seed = st.sidebar.number_input("Price-world seed", value=DEFAULT_CONFIG.seed, step=1)
config = DEFAULT_CONFIG.with_overrides(seed=int(seed))
if st.sidebar.button("Save charts as PNG"):
    paths = save_dashboard_pngs(config=config)
    st.sidebar.success("Saved:\n" + "\n".join(p.name for p in paths))


@st.cache_data(show_spinner=False)
def _load(seed_value: int):
    cfg = DEFAULT_CONFIG.with_overrides(seed=seed_value)
    data = build_dashboard(cfg)
    gen = default_verdict_generator()
    verdicts = {aid: gen.verdict(s) for aid, s in data.scores_by_id.items()}
    return data, verdicts


data, verdicts = _load(int(seed))

# 1. LEADERBOARD ---------------------------------------------------------------------
st.header("Leaderboard — ranked by Beat-the-Market")
st.dataframe(
    leaderboard_dataframe(data.leaderboard),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Beat-Market": st.column_config.NumberColumn(format="%.2f%%", help="Mean excess return vs index"),
        "Direction Hit-Rate": st.column_config.NumberColumn(format="%.0f%%"),
        "Accuracy": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.pyplot(plot_leaderboard(data.leaderboard))

# 2. PROFILE -------------------------------------------------------------------------
st.header("Analyst Profile")
names = {s.analyst_name: aid for aid, s in data.scores_by_id.items()}
ordered_names = [r.analyst_name for r in data.leaderboard.rows]
choice = st.selectbox("Choose an analyst", ordered_names)
aid = names[choice]
score = data.scores_by_id[aid]

st.info(f"**Verdict:** {verdicts[aid]}")
col1, col2, col3 = st.columns(3)
col1.metric("Beat-the-Market", "—" if score.beat_market is None else f"{score.beat_market*100:+.1f}%")
col2.metric("Direction Hit-Rate", f"{score.direction_hit_rate*100:.0f}%")
col3.metric("Accuracy", "—" if score.mean_accuracy is None else f"{score.mean_accuracy:.3f}")

st.pyplot(plot_analyst_profile(score))

# 3. DRILL-DOWN ----------------------------------------------------------------------
st.header("Call-level drill-down — full traceability")
st.caption("Every score traces to the exact call and the prices that resolved it.")
st.dataframe(call_detail_dataframe(score), use_container_width=True, hide_index=True)
