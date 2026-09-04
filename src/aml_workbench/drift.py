"""Drift monitoring: Population Stability Index on model scores and features.

Reference distribution = training period (steps 1-34), compared per monitored
column against the test period (35-49). A breach above the configured PSI
threshold flags the column and fails the run closed (non-zero exit), so drift
triggers review instead of passing silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from aml_workbench import config
from aml_workbench.errors import DataQualityError
from aml_workbench.model import load_labeled, split_temporal


def psi(expected: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    """Population Stability Index: sum over bins of (a - e) * ln(a / e).

    Quantile bins come from the reference (expected) distribution; empty-bin
    proportions are floored at config.PSI_ZERO_EPS to keep ln finite.
    """
    # interior quantile entries only (probability 0 and 1 dropped), padded with
    # infinite extremes: padding first/last UNIQUE values would delete a real
    # cutpoint whenever the extreme quantiles repeat an interior value
    interior = np.unique(
        np.quantile(expected, np.linspace(0, 1, config.PSI_BINS + 1), method="inverted_cdf")[1:-1]
    )
    if expected.min() == expected.max():
        # constant reference: cut just above the constant value so the
        # reference occupies one bin and any shift up OR down registers —
        # returning 0 here would mask real drift
        interior = np.array([expected[0], np.nextafter(expected[0], np.inf)])
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    e = np.histogram(expected, bins=edges)[0] / expected.size
    a = np.histogram(actual, bins=edges)[0] / actual.size
    e = np.clip(e, config.PSI_ZERO_EPS, None)
    a = np.clip(a, config.PSI_ZERO_EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run_drift(data_dir: Path) -> tuple[str, bool]:
    """PSI per feature column + on model scores, train vs test period. Writes
    the drift report; returns (summary, breached). Fail-closed on missing
    artifacts."""
    model_path = data_dir / "models" / "challenger.joblib"
    if not model_path.exists():
        raise DataQualityError("missing models/challenger.joblib - run `aml challenger` first")
    x, y, steps, names = load_labeled(data_dir / "workbench.duckdb", include_graph=True)
    train_mask, test_mask = split_temporal(steps)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    train_scores = model.predict_proba(x[train_mask])[:, 1]
    test_scores = model.predict_proba(x[test_mask])[:, 1]

    columns: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(names):
        columns[name] = _psi_entry(x[train_mask, i], x[test_mask, i])
    columns["score"] = _psi_entry(train_scores, test_scores)

    breached = any(bool(c["breach"]) for c in columns.values())
    watched = [n for n, c in columns.items() if not c["breach"] and c["watch"]]

    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": "training steps 1-34",
        "comparison": "test steps 35-49",
        "thresholds": {"breach": config.PSI_BREACH_THRESHOLD, "watch": config.PSI_WATCH_THRESHOLD},
        "breached": breached,
        "watch": sorted(watched),
        "columns": columns,
    }
    (report_dir / "drift_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (report_dir / "drift_report.md").write_text(_render_report(payload), encoding="utf-8")
    breaches = sorted(n for n, c in columns.items() if c["breach"])
    summary = (
        f"drift: {len(breaches)} breach(es) [{', '.join(breaches) if breaches else 'none'}], "
        f"{len(watched)} on watchlist -> {report_dir / 'drift_report.md'}"
    )
    return summary, breached


def _psi_entry(train: NDArray[np.float64], test: NDArray[np.float64]) -> dict[str, Any]:
    value = psi(train, test)
    return {
        "psi": value,
        "breach": bool(value > config.PSI_BREACH_THRESHOLD),
        "watch": bool(value > config.PSI_WATCH_THRESHOLD),
    }


def _render_report(payload: dict[str, Any]) -> str:
    columns: dict[str, dict[str, Any]] = payload["columns"]
    lines = [
        "# PSI drift monitoring",
        "",
        f"Reference: {payload['reference']} vs {payload['comparison']}. "
        f"Breach threshold {payload['thresholds']['breach']}, "
        f"watch threshold {payload['thresholds']['watch']}.",
        "",
        "| column | PSI | status |",
        "|---|---|---|",
    ]
    for name, c in sorted(columns.items(), key=lambda kv: -float(kv[1]["psi"])):
        status = "BREACH" if c["breach"] else ("watch" if c["watch"] else "ok")
        lines.append(f"| {name} | {c['psi']:.4f} | {status} |")
    lines.append("")
    return "\n".join(lines)
