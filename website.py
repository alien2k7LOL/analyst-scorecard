"""Analyst Scorecard — plain locally-hosted website (no Streamlit).

Builds a single self-contained ``site/index.html`` from the SAME validated engine the
CLIs use (synthetic dashboard + historical back-test), then serves it over plain HTTP.

    .venv/bin/python website.py            # build site/ and serve at http://localhost:8000
    .venv/bin/python website.py --build    # just write site/index.html, don't serve
    .venv/bin/python website.py --port 9000

Why this exists: it's an ordinary static website — inline CSS/JS, charts embedded as PNGs,
served by Python's built-in http.server. No app framework, no live dependency at view time.
Everything is rendered once at build; the page itself just needs a browser.

Offline-first: no network, no API key required (verdicts fall back to the templated generator).
The scoring engine is reused UNCHANGED — this file only renders its output as HTML.
"""

from __future__ import annotations

import argparse
import base64
import functools
import html
import io
import http.server
import socketserver
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display required
import matplotlib.pyplot as plt

from analyst_scorecard.backtest import SAMPLE_DATA_DIR, run_backtest
from analyst_scorecard.config import DEFAULT_CONFIG
from analyst_scorecard.verdicts import default_verdict_generator
from analyst_scorecard.viz import (
    build_dashboard,
    call_detail_dataframe,
    plot_analyst_profile,
    plot_leaderboard,
)

SITE_DIR = Path(__file__).resolve().parent / "site"


# --------------------------------------------------------------------------------------
# Small HTML helpers
# --------------------------------------------------------------------------------------


def _fig_to_img(fig, *, alt: str = "") -> str:
    """Render a matplotlib figure to an inline <img> (base64 PNG) and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img class="chart" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def _pct(x, digits: int = 1, signed: bool = True) -> str:
    if x is None:
        return "—"
    fmt = f"{{:+.{digits}f}}%" if signed else f"{{:.{digits}f}}%"
    return fmt.format(x * 100)


def _num(x, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _cls_for(beat) -> str:
    if beat is None:
        return "neutral"
    return "pos" if beat > 0 else "neg"


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _leaderboard_html(leaderboard, verdicts) -> str:
    head = (
        "<tr><th>#</th><th>Analyst</th><th>Firm</th>"
        "<th class='r'>Beat-Market</th><th class='r'>Dir. Hit-Rate</th>"
        "<th class='r'>Accuracy</th><th class='r'>Calls</th><th class='r'>Directional</th></tr>"
    )
    body = []
    for rank, s in enumerate(leaderboard.rows, start=1):
        verdict = verdicts.get(s.analyst_id, "")
        body.append(
            "<tr>"
            f"<td class='rank'>{rank}</td>"
            f"<td class='name'>{html.escape(s.analyst_name)}</td>"
            f"<td class='firm'>{html.escape(s.firm)}</td>"
            f"<td class='r {_cls_for(s.beat_market)}'>{_pct(s.beat_market)}</td>"
            f"<td class='r'>{_pct(s.direction_hit_rate, digits=0)}</td>"
            f"<td class='r'>{_num(s.mean_accuracy)}</td>"
            f"<td class='r'>{s.n_calls}</td>"
            f"<td class='r'>{s.n_directional}</td>"
            "</tr>"
            f"<tr class='verdict'><td></td><td colspan='7'>{html.escape(verdict)}</td></tr>"
        )
    return f"<table class='lb'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def _analyst_blocks_html(leaderboard, scores_by_id, verdicts, prefix: str) -> str:
    """A <select> plus one hidden <div> per analyst (chart + drill-down)."""
    options = []
    blocks = []
    for i, row in enumerate(leaderboard.rows):
        aid = row.analyst_id
        score = scores_by_id[aid]
        selected = " selected" if i == 0 else ""
        display = "block" if i == 0 else "none"
        options.append(
            f"<option value='{html.escape(aid)}'{selected}>{html.escape(row.analyst_name)}</option>"
        )

        chart = _fig_to_img(plot_analyst_profile(score), alt=f"{row.analyst_name} profile")
        tiles = (
            "<div class='tiles'>"
            f"<div class='tile'><span class='k'>Beat-the-Market</span>"
            f"<span class='v {_cls_for(score.beat_market)}'>{_pct(score.beat_market)}</span></div>"
            f"<div class='tile'><span class='k'>Direction Hit-Rate</span>"
            f"<span class='v'>{_pct(score.direction_hit_rate, digits=0)}</span></div>"
            f"<div class='tile'><span class='k'>Accuracy</span>"
            f"<span class='v'>{_num(score.mean_accuracy)}</span></div>"
            "</div>"
        )
        verdict = f"<p class='verdict-line'>{html.escape(verdicts.get(aid, ''))}</p>"
        table = call_detail_dataframe(score).to_html(index=False, border=0, classes="drill")

        blocks.append(
            f"<div class='analyst {prefix}-analyst' id='{prefix}-{html.escape(aid)}' style='display:{display}'>"
            f"{verdict}{tiles}{chart}"
            f"<h4>Call-level drill-down — full traceability</h4>"
            f"<div class='scroll'>{table}</div>"
            "</div>"
        )

    select = (
        f"<label class='picker'>Choose an analyst "
        f"<select onchange=\"showAnalyst('{prefix}', this.value)\">{''.join(options)}</select>"
        "</label>"
    )
    return select + "".join(blocks)


def _skips_html(result) -> str:
    if not (result.skipped or result.ingest_issues):
        return ""
    parts = ["<details class='skips'><summary>Skipped &amp; dropped calls "
             "(transparency — never silently scored)</summary>"]
    if result.skip_reason_counts:
        parts.append("<p><b>Skipped at resolution:</b> " +
                     ", ".join(f"{html.escape(k)} × {v}" for k, v in result.skip_reason_counts.items()) + "</p>")
    if result.ingest_reason_counts:
        parts.append("<p><b>Dropped at ingest:</b> " +
                     ", ".join(f"{html.escape(k)} × {v}" for k, v in result.ingest_reason_counts.items()) + "</p>")
    if result.skipped:
        rows = "".join(
            f"<tr><td>{html.escape(s.call.call_id)}</td><td>{html.escape(s.call.ticker)}</td>"
            f"<td>{html.escape(s.reason)}</td><td>{html.escape(str(s.detail))}</td></tr>"
            for s in result.skipped
        )
        parts.append("<table class='drill'><thead><tr><th>call_id</th><th>ticker</th>"
                     f"<th>reason</th><th>detail</th></tr></thead><tbody>{rows}</tbody></table>")
    if result.ingest_issues:
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(iss.get(c, '')))}</td>"
                             for c in ("call_id", "ticker", "reason", "detail")) + "</tr>"
            for iss in result.ingest_issues
        )
        parts.append("<table class='drill'><thead><tr><th>call_id</th><th>ticker</th>"
                     f"<th>reason</th><th>detail</th></tr></thead><tbody>{rows}</tbody></table>")
    parts.append("</details>")
    return "".join(parts)


# --------------------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------------------

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#6b7280;--line:#e5e7eb;
--pos:#1a9850;--neg:#d73027;--accent:#2b6cb0;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 22px 80px}
header h1{margin:0 0 4px;font-size:26px}
header p{margin:0;color:var(--muted);max-width:760px}
.tabbar{display:flex;gap:6px;margin:22px 0 0;border-bottom:2px solid var(--line)}
.tabbtn{appearance:none;border:0;background:none;padding:10px 16px;font-size:15px;cursor:pointer;
color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-2px}
.tabbtn.active{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.tab{display:none;padding-top:8px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin:18px 0;box-shadow:0 1px 2px rgba(16,24,40,.04)}
h2{font-size:18px;margin:0 0 12px}
h4{font-size:14px;margin:18px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.meta{color:var(--muted);font-size:13px;margin:0 0 10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
.lb th,.lb td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}
.lb th.r,.lb td.r{text-align:right;font-variant-numeric:tabular-nums}
.lb .rank{color:var(--muted);width:34px}
.lb .name{font-weight:600}.lb .firm{color:var(--muted)}
.lb .verdict td{border-bottom:1px solid var(--line);color:var(--muted);font-size:12.5px;
padding:0 10px 9px 10px}
.pos{color:var(--pos);font-weight:600}.neg{color:var(--neg);font-weight:600}.neutral{color:var(--muted)}
.chart{display:block;max-width:100%;height:auto;margin:14px auto;border-radius:8px}
.picker{display:inline-block;margin:6px 0 4px;font-size:14px;color:var(--muted)}
.picker select{font-size:14px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;margin-left:6px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.tile{flex:1 1 150px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.tile .k{display:block;color:var(--muted);font-size:12px}
.tile .v{display:block;font-size:22px;font-weight:700;margin-top:2px}
.verdict-line{background:#eef5fb;border-left:3px solid var(--accent);padding:10px 12px;border-radius:6px;margin:6px 0 2px}
.scroll{overflow-x:auto}
.drill{font-size:12px}.drill th,.drill td{padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.drill thead th{position:sticky;top:0;background:var(--card)}
details.skips{margin-top:14px}summary{cursor:pointer;color:var(--accent);font-weight:600}
footer{color:var(--muted);font-size:12.5px;margin-top:26px;text-align:center}
"""

_JS = """
function showTab(t){
  document.querySelectorAll('.tab').forEach(e=>e.style.display='none');
  document.getElementById('tab-'+t).style.display='block';
  document.querySelectorAll('.tabbtn').forEach(b=>b.classList.remove('active'));
  document.getElementById('btn-'+t).classList.add('active');
}
function showAnalyst(prefix,id){
  document.querySelectorAll('.'+prefix+'-analyst').forEach(e=>e.style.display='none');
  document.getElementById(prefix+'-'+id).style.display='block';
}
"""


def build_html(data_dir: Path = SAMPLE_DATA_DIR) -> str:
    gen = default_verdict_generator()

    # ---- synthetic ----
    dash = build_dashboard(DEFAULT_CONFIG)
    syn_verdicts = {aid: gen.verdict(s) for aid, s in dash.scores_by_id.items()}
    syn_lb = _leaderboard_html(dash.leaderboard, syn_verdicts)
    syn_bar = _fig_to_img(plot_leaderboard(dash.leaderboard), alt="Synthetic leaderboard")
    syn_blocks = _analyst_blocks_html(dash.leaderboard, dash.scores_by_id, syn_verdicts, "syn")

    # ---- historical ----
    result = run_backtest(data_dir)
    hist_verdicts = {s.analyst_id: gen.verdict(s) for s in result.leaderboard.rows}
    hist_lb = _leaderboard_html(result.leaderboard, hist_verdicts)
    hist_bar = _fig_to_img(plot_leaderboard(result.leaderboard), alt="Historical leaderboard")
    hist_blocks = _analyst_blocks_html(result.leaderboard, result.analyst_scores, hist_verdicts, "hist")
    tag = ("SAMPLE data — synthetic & fictional (replace with your own files)"
           if result.is_sample else "user-supplied data")
    span = (f"Price span {result.span_start} → {result.span_end}"
            if result.span_start else "")
    counts = (f"<b>{result.n_resolved}</b> resolved &amp; scored · "
              f"<b>{result.n_skipped}</b> skipped at resolution · "
              f"<b>{result.n_ingest_dropped}</b> dropped at ingest")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analyst Scorecard</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>📊 Analyst Scorecard</h1>
  <p>Honest, fair, reproducible grading of analyst price targets. The headline metric is
  <b>beat-the-market</b>: would you have done better just buying the index?</p>
</header>

<div class="tabbar">
  <button class="tabbtn active" id="btn-syn" onclick="showTab('syn')">🧪 Synthetic engine demo</button>
  <button class="tabbtn" id="btn-hist" onclick="showTab('hist')">📜 Historical back-test</button>
</div>

<div class="tab" id="tab-syn" style="display:block">
  <section>
    <h2>Leaderboard — ranked by Beat-the-Market</h2>
    {syn_lb}{syn_bar}
  </section>
  <section>
    <h2>Analyst profile &amp; drill-down</h2>
    <p class="meta">Above the diagonal beats the index; below it lagged. Every score traces to the
    exact call and the prices that resolved it.</p>
    {syn_blocks}
  </section>
</div>

<div class="tab" id="tab-hist">
  <section>
    <h2>Historical Leaderboard</h2>
    <p class="meta">Source: <b>{html.escape(tag)}</b>{(' · ' + span) if span else ''}<br>{counts}</p>
    {hist_lb}{hist_bar}
    {_skips_html(result)}
  </section>
  <section>
    <h2>Historical analyst profile &amp; drill-down</h2>
    <p class="meta">Each historical call resolved using ONLY prices from its original window
    [call date → horizon] — no look-ahead, ever.</p>
    {hist_blocks}
  </section>
</div>

<footer>Built offline from the same look-ahead-safe scoring engine the CLIs use ·
no network, no API key required.</footer>
</div>
<script>{_JS}</script>
</body></html>"""


def build_site(data_dir: Path = SAMPLE_DATA_DIR) -> Path:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(build_html(data_dir), encoding="utf-8")
    return out


def serve(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE_DIR))
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n  Analyst Scorecard is live at  {url}")
        print("  Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Stopped.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build & serve the Analyst Scorecard website.")
    p.add_argument("--data-dir", default=str(SAMPLE_DATA_DIR), help="historical data folder")
    p.add_argument("--port", type=int, default=8000, help="port to serve on (default 8000)")
    p.add_argument("--build", action="store_true", help="build site/index.html and exit (don't serve)")
    args = p.parse_args(argv)

    out = build_site(Path(args.data_dir))
    print(f"  Built {out}  ({out.stat().st_size // 1024} KB)")
    if not args.build:
        serve(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
