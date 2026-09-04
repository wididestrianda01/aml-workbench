"""Validation: strict-inductive expanding-window walk-forward over the locked
temporal split, with per-timestep metrics and the illicit base-rate curve.

Protocol: for each test step t (35-49) the model trains on labeled rows
strictly before t and evaluates on the rows AT t. Combined with the graph
stage's point-in-time feature protocol, no future adjacency or label can leak
into training or feature computation. The split boundary is asserted, not
assumed: a training row at or after the as-of step fails the run closed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from aml_workbench import config
from aml_workbench.errors import DataQualityError
from aml_workbench.model import load_labeled

TrainMaskFn = Callable[[NDArray[np.int64], int], NDArray[np.bool_]]


def train_mask_before(steps: NDArray[np.int64], as_of_step: int) -> NDArray[np.bool_]:
    """Strict-inductive training mask: labeled rows strictly before the as-of
    step. Never random; the as-of step itself is excluded."""
    return steps < as_of_step


def _assert_no_lookahead(
    steps: NDArray[np.int64], mask: NDArray[np.bool_], as_of_step: int
) -> None:
    """Split-boundary assertion: the split function is injected, so it is
    verified rather than trusted — a leaky split fails the run closed."""
    train_steps = steps[mask]
    if train_steps.size == 0 or train_steps.max() >= as_of_step:
        raise DataQualityError(
            f"look-ahead detected: training rows at or after as-of step {as_of_step}"
        )


def run_validation(data_dir: Path, *, train_mask_fn: TrainMaskFn = train_mask_before) -> str:
    """Expanding-window walk-forward; writes the per-timestep artifact + the
    base-rate curve. Fail-closed on missing tables."""
    db_path = data_dir / "workbench.duckdb"
    x, y, steps, _feature_names = load_labeled(db_path, include_graph=True)

    # base-rate curve: illicit share of labeled rows per step (all 49 steps)
    curve: list[dict[str, Any]] = [
        {
            "step": int(s),
            "n": int((steps == s).sum()),
            "illicit": int(((steps == s) & (y == 1)).sum()),
            "illicit_rate": float(y[steps == s].mean()),
        }
        for s in range(1, int(steps.max()) + 1)
    ]

    per_step: list[dict[str, Any]] = []
    for as_of in sorted({int(s) for s in steps if s >= config.TEST_STEP_MIN}):
        mask = train_mask_fn(steps, as_of)
        _assert_no_lookahead(steps, mask, as_of)
        eval_mask = steps == as_of
        model = lgb.LGBMClassifier(
            n_estimators=config.LGBM_N_ESTIMATORS,
            learning_rate=config.LGBM_LEARNING_RATE,
            num_leaves=config.LGBM_NUM_LEAVES,
            class_weight="balanced",
            random_state=config.MODEL_SEEDS[0],
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x[mask], y[mask])
        scores = np.asarray(model.predict_proba(x[eval_mask]))[:, 1]
        preds = (scores >= config.VALIDATION_F1_THRESHOLD).astype(int)
        y_eval = y[eval_mask]
        per_step.append(
            {
                "step": as_of,
                "n_eval": int(eval_mask.sum()),
                "illicit_rate": float(y_eval.mean()),
                "f1": float(f1_score(y_eval, preds, zero_division=0)),
                "precision": float(precision_score(y_eval, preds, zero_division=0)),
                "recall": float(recall_score(y_eval, preds, zero_division=0)),
                "pr_auc": float(average_precision_score(y_eval, scores)),
            }
        )

    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "strict-inductive expanding-window walk-forward (train < as-of step)",
        "seed": config.MODEL_SEEDS[0],
        "threshold": config.VALIDATION_F1_THRESHOLD,
        "seed_rationale": (
            "one fixed seed per walk-forward refit: the >=3-seed protocol is "
            "applied at challenger selection (3 seeds x 15 refits here would "
            "multiply cost with no selection riding on this stage); metrics "
            "are per-step observations, not a seed-averaged decision"
        ),
        "per_step": per_step,
        "base_rate_curve": curve,
    }
    (report_dir / "validation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (report_dir / "validation_report.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    f1s = [float(m["f1"]) for m in per_step]
    return (
        f"validate: walk-forward over {len(per_step)} test steps, "
        f"mean F1 {float(np.mean(f1s)):.4f} -> {report_dir / 'validation_report.md'}"
    )


def _render_report(payload: dict[str, Any]) -> str:
    per_step: list[dict[str, Any]] = payload["per_step"]
    curve: list[dict[str, Any]] = payload["base_rate_curve"]
    lines = [
        "# Walk-forward validation (strict-inductive, expanding window)",
        "",
        f"Protocol: {payload['protocol']}. Seed {payload['seed']}, "
        f"decision threshold {payload['threshold']}.",
        "",
        "## Per-timestep metrics (test steps)",
        "",
        "| step | n | illicit rate | F1 | precision | recall | PR-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in per_step:
        lines.append(
            f"| {m['step']} | {m['n_eval']} | {m['illicit_rate']:.3f} "
            f"| {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} "
            f"| {m['pr_auc']:.3f} |"
        )
    lines += [
        "",
        "## Micro-F1 caveat",
        "",
        "Aggregate (micro) F1 on this split is dominated by the majority class:",
        "with illicit rates this low, a trivial all-licit predictor scores roughly",
        "the majority share (~0.9+). A micro-F1 near 0.98 is NOT detection — read",
        "the per-timestep F1/precision/recall above, which are computed on each",
        "step's illicit minority.",
        "",
        "## Illicit base-rate curve",
        "",
        "| step | illicit rate |",
        "|---|---|",
    ]
    for c in curve:
        lines.append(f"| {c['step']} | {c['illicit_rate']:.4f} |")
    lines.append("")
    return "\n".join(lines)
