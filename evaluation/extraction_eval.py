"""Audit the recommendation extractor: confusion matrices, per-class metrics, field accuracy,
error report, failure breakdown, and figures. Pure measurement — production code is untouched.

    python -m evaluation.extraction_eval        # prints tables, writes figures + REPORT.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyst_scorecard.intel.extract import extract_recommendation
from analyst_scorecard.schemas import RATING_TO_DIRECTION, Rating

from .gold_set import GOLD, GoldExample

FIG_DIR = Path(__file__).resolve().parent / "figures"
RATING_CLASSES = ["Buy", "Overweight", "Hold", "Underweight", "Sell", "None"]
DIRECTION_CLASSES = ["UP", "DOWN", "FLAT", "None"]
FIELDS = ["ticker", "rating", "target_price", "analyst", "firm", "date"]


# --------------------------------------------------------------------------------------
# Run the extractor over the gold set (heuristic only — deterministic, offline, audit-stable)
# --------------------------------------------------------------------------------------


@dataclass
class Pair:
    gold: GoldExample
    pred_ticker: Optional[str]
    pred_rating: Optional[str]
    pred_target: Optional[float]
    pred_analyst: Optional[str]
    pred_firm: Optional[str]
    pred_date: Optional[str]
    source: str


def run_extractions(gold: list[GoldExample] = GOLD, *, use_llm: bool = False) -> list[Pair]:
    pairs: list[Pair] = []
    for ex in gold:
        r = extract_recommendation(ex.text, use_llm=use_llm)
        pairs.append(Pair(
            gold=ex, pred_ticker=r.ticker, pred_rating=r.rating, pred_target=r.target_price,
            pred_analyst=r.analyst, pred_firm=r.firm,
            pred_date=r.publication_date.isoformat() if r.publication_date else None, source=r.source,
        ))
    return pairs


def _field(pair: Pair, field: str):
    """(expected, predicted) for a field, normalized for comparison."""
    g = pair.gold
    return {
        "ticker": (g.ticker, pair.pred_ticker),
        "rating": (g.rating, pair.pred_rating),
        "target_price": (g.target_price, pair.pred_target),
        "analyst": (g.analyst, pair.pred_analyst),
        "firm": (g.firm, pair.pred_firm),
        "date": (g.date, pair.pred_date),
    }[field]


# --------------------------------------------------------------------------------------
# Confusion matrices
# --------------------------------------------------------------------------------------


def _to_direction(rating: Optional[str]) -> str:
    if rating is None:
        return "None"
    return RATING_TO_DIRECTION[Rating(rating)].value.upper()


def confusion_matrix(pairs: list[Pair], mode: str = "rating") -> pd.DataFrame:
    """Rows = actual, columns = predicted. ``mode`` is 'rating' (5-class) or 'direction' (UP/DOWN/FLAT).

    The 'direction' matrix is the synonym-normalized view (Overweight→UP/Buy-side, Underweight→DOWN).
    """
    classes = RATING_CLASSES if mode == "rating" else DIRECTION_CLASSES
    cm = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for p in pairs:
        if mode == "rating":
            a, b = (p.gold.rating or "None"), (p.pred_rating or "None")
        else:
            a, b = _to_direction(p.gold.rating), _to_direction(p.pred_rating)
        cm.loc[a, b] += 1
    cm.index.name = "actual ↓ / predicted →"
    return cm


def classification_report(cm: pd.DataFrame) -> pd.DataFrame:
    """Per-class precision/recall/F1/support + accuracy and macro/weighted averages, from a matrix."""
    classes = list(cm.index)
    total = int(cm.values.sum())
    rows = {}
    for c in classes:
        tp = int(cm.loc[c, c])
        support = int(cm.loc[c, :].sum())
        pred_pos = int(cm.loc[:, c].sum())
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows[c] = dict(precision=precision, recall=recall, f1=f1, support=support)
    df = pd.DataFrame(rows).T[["precision", "recall", "f1", "support"]]

    present = df[df["support"] > 0]
    accuracy = sum(int(cm.loc[c, c]) for c in classes) / total if total else 0.0
    macro = present[["precision", "recall", "f1"]].mean()
    w = present["support"] / present["support"].sum()
    weighted = (present[["precision", "recall", "f1"]].T * w).T.sum()

    df.loc["accuracy"] = [np.nan, np.nan, accuracy, total]
    df.loc["macro avg"] = [macro["precision"], macro["recall"], macro["f1"], total]
    df.loc["weighted avg"] = [weighted["precision"], weighted["recall"], weighted["f1"], total]
    return df.round(3)


# --------------------------------------------------------------------------------------
# Field-level accuracy
# --------------------------------------------------------------------------------------


def field_accuracy(pairs: list[Pair]) -> pd.DataFrame:
    rows = {}
    for field in FIELDS:
        correct = present_correct = present_total = 0
        for p in pairs:
            exp, got = _field(p, field)
            ok = exp == got
            correct += int(ok)
            if exp is not None:
                present_total += 1
                present_correct += int(ok)
        rows[field] = dict(
            accuracy=correct / len(pairs),
            n=len(pairs),
            present=present_total,
            accuracy_when_present=(present_correct / present_total) if present_total else float("nan"),
        )
    return pd.DataFrame(rows).T[["accuracy", "accuracy_when_present", "present", "n"]].round(3)


# --------------------------------------------------------------------------------------
# Error report + failure breakdown
# --------------------------------------------------------------------------------------


def _wrong_fields(p: Pair) -> list[str]:
    return [f for f in FIELDS if _field(p, f)[0] != _field(p, f)[1]]


def error_report(pairs: list[Pair]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        wrong = _wrong_fields(p)
        if not wrong:
            continue
        rows.append({
            "text": p.gold.text,
            "wrong_fields": ", ".join(wrong),
            "n_wrong": len(wrong),
            "expected": {f: _field(p, f)[0] for f in wrong},
            "predicted": {f: _field(p, f)[1] for f in wrong},
            "source": p.source,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_wrong", ascending=False).reset_index(drop=True)
    return df


def _categorize(field: str, exp, got) -> Optional[str]:
    if exp == got:
        return None
    if field in ("ticker", "target_price", "rating"):
        noun = {"ticker": "ticker", "target_price": "target", "rating": "rating"}[field]
        if exp is not None and got is None:
            return f"missing {noun}"
        if exp is None and got is not None:
            return f"spurious {noun}"
        return ("rating confusion" if field == "rating" else f"wrong {noun}")
    return {"date": "date parsing", "analyst": "analyst extraction", "firm": "firm extraction"}[field]


def failure_breakdown(pairs: list[Pair]) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for p in pairs:
        for f in FIELDS:
            exp, got = _field(p, f)
            cat = _categorize(f, exp, got)
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
    total = sum(counts.values())
    rows = [{"failure_type": k, "count": v, "pct_of_errors": round(100 * v / total, 1) if total else 0.0}
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def save_confusion_heatmap(cm: pd.DataFrame, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    data = cm.values.astype(float)
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), labels=cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), labels=cm.index)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(title)
    thresh = data.max() / 2 if data.max() else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, int(data[i, j]), ha="center", va="center",
                    color="white" if data[i, j] > thresh else "#222222", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_bar(series: pd.Series, path: Path, title: str, ylabel: str, color: str = "#4c9be8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(range(len(series)), series.values, color=color)
    ax.set_xticks(range(len(series)), labels=series.index, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    for i, v in enumerate(series.values):
        ax.text(i, v, f"{v:.2f}" if v < 2 else f"{int(v)}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_figures(pairs: list[Pair], figdir: Path = FIG_DIR) -> dict[str, Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    cm_r = confusion_matrix(pairs, "rating")
    cm_d = confusion_matrix(pairs, "direction")
    fa = field_accuracy(pairs)["accuracy"]
    fb = failure_breakdown(pairs)
    paths = {
        "rating_confusion": save_confusion_heatmap(cm_r, figdir / "rating_confusion.png", "Rating confusion (extractor)"),
        "direction_confusion": save_confusion_heatmap(cm_d, figdir / "direction_confusion.png", "Direction confusion (extractor)"),
        "field_accuracy": save_bar(fa, figdir / "field_accuracy.png", "Field-level extraction accuracy", "accuracy"),
    }
    if not fb.empty:
        paths["failure_breakdown"] = save_bar(
            fb.set_index("failure_type")["count"], figdir / "failure_breakdown.png",
            "Failure categories", "count", color="#d9785a")
    return paths


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def main() -> int:
    pairs = run_extractions()
    cm_rating = confusion_matrix(pairs, "rating")
    cm_dir = confusion_matrix(pairs, "direction")
    rep_rating = classification_report(cm_rating)
    rep_dir = classification_report(cm_dir)
    fields = field_accuracy(pairs)
    errors = error_report(pairs)
    failures = failure_breakdown(pairs)
    paths = generate_figures(pairs)

    out = []
    out.append(f"# Extraction evaluation — {len(pairs)} gold examples (heuristic parser)\n")
    out.append("## Rating confusion matrix (5-class)\n```\n" + cm_rating.to_string() + "\n```\n")
    out.append("## Rating metrics\n```\n" + rep_rating.to_string() + "\n```\n")
    out.append("## Direction confusion matrix (normalized: UP/DOWN/FLAT)\n```\n" + cm_dir.to_string() + "\n```\n")
    out.append("## Direction metrics\n```\n" + rep_dir.to_string() + "\n```\n")
    out.append("## Field-level accuracy\n```\n" + fields.to_string() + "\n```\n")
    out.append("## Failure breakdown\n```\n" + failures.to_string(index=False) + "\n```\n")
    out.append(f"## Errors ({len(errors)} of {len(pairs)} examples had ≥1 wrong field)\n")
    for _, r in errors.iterrows():
        out.append(f"- [{r['wrong_fields']}] (src={r['source']})\n  text: {r['text']}\n"
                   f"  expected {r['expected']}  →  predicted {r['predicted']}")
    out.append("\n## Figures\n" + "\n".join(f"- {k}: {v}" for k, v in paths.items()))
    report = "\n".join(out)

    (Path(__file__).resolve().parent / "REPORT.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
