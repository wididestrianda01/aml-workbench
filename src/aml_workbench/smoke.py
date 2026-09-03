"""C5 — smoke run: LR + RF on the strict temporal split, gated, with report.

Trains on labeled steps 1-34, tests on labeled steps 35-49 (temporal split
only, never random; unknown-class nodes excluded from training and scoring).
Gates at ROC-AUC >= 0.80 and < 10 minutes wall clock — below/over the gate the
run exits non-zero and NO report is written (fail-closed). PR-AUC is reported
from the start: it is the predeclared challenger metric in Phase 4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from aml_workbench import config
from aml_workbench.errors import DataQualityError, SmokeGateError
from aml_workbench.report import render_smoke_report


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class SmokeResult:
    models: list[ModelMetrics]
    train_base_rate: float
    test_base_rate: float
    train_rows: int
    test_rows: int
    runtime_s: float
    gate_threshold: float
    passed: bool


def _load_labeled(db_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labeled rows only: X (n, 166), y (illicit=1), steps — ordered by tx_id."""
    feature_cols = ", ".join(f"f{i:03d}" for i in range(1, config.FEATURE_COUNT + 1))
    query = f"""
        SELECT f.time_step, t.class_label, {feature_cols}
        FROM elliptic_tx_features f
        JOIN elliptic_tx t USING (tx_id)
        WHERE t.class_label IS NOT NULL
        ORDER BY f.tx_id
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        required = {"elliptic_tx", "elliptic_tx_features"}
        if not required <= tables:
            raise DataQualityError(
                f"DuckDB store {db_path} lacks Elliptic ingest tables {sorted(required)}; "
                "run 'aml ingest' first."
            )
        rows = con.execute(query).fetchnumpy()
    finally:
        con.close()

    steps = np.asarray(rows["time_step"], dtype=np.int64)
    class_label = np.asarray(rows["class_label"], dtype=np.int64)
    y = (class_label == 1).astype(np.int64)
    feature_names = [f"f{i:03d}" for i in range(1, config.FEATURE_COUNT + 1)]
    x = np.column_stack([np.asarray(rows[name], dtype=np.float64) for name in feature_names])
    if np.isnan(x).any():
        raise DataQualityError("NULL/NaN feature values found in the labeled smoke set.")
    return x, y, steps


def _split_temporal(steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locked temporal split: train steps 1-34, test steps 35-49. Never random."""
    train_mask = steps <= config.TRAIN_STEP_MAX
    test_mask = steps >= config.TEST_STEP_MIN
    if not train_mask.any() or not test_mask.any():
        raise DataQualityError("Temporal split produced an empty train or test side.")
    return train_mask, test_mask


def _fit_and_score(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> list[ModelMetrics]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_test = scaler.transform(x[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]

    models = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=config.SMOKE_SEED,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.SMOKE_SEED,
            ),
        ),
    ]
    metrics: list[ModelMetrics] = []
    for name, model in models:
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        metrics.append(
            ModelMetrics(
                name=name,
                roc_auc=float(roc_auc_score(y_test, scores)),
                pr_auc=float(average_precision_score(y_test, scores)),
            )
        )
    return metrics


def run_smoke(data_dir: Path) -> Path:
    """Run the C5 smoke gate; return the report path. Fail-closed: below the
    gate (or over the runtime limit) raises SmokeGateError and writes nothing."""
    db_path = data_dir / "workbench.duckdb"
    started = time.monotonic()
    x, y, steps = _load_labeled(db_path)
    train_mask, test_mask = _split_temporal(steps)
    metrics = _fit_and_score(x, y, train_mask, test_mask)
    runtime_s = time.monotonic() - started

    best_roc = max(m.roc_auc for m in metrics)
    passed = best_roc >= config.SMOKE_ROC_AUC_GATE and runtime_s <= config.SMOKE_RUNTIME_LIMIT_S
    result = SmokeResult(
        models=metrics,
        train_base_rate=float(y[train_mask].mean()),
        test_base_rate=float(y[test_mask].mean()),
        train_rows=int(train_mask.sum()),
        test_rows=int(test_mask.sum()),
        runtime_s=runtime_s,
        gate_threshold=config.SMOKE_ROC_AUC_GATE,
        passed=passed,
    )
    if not passed:
        # Fail-closed: below the gate no report artifact exists.
        raise SmokeGateError(
            f"Smoke gate failed: best ROC-AUC {best_roc:.4f} vs threshold "
            f"{config.SMOKE_ROC_AUC_GATE}, runtime {runtime_s:.1f}s vs limit "
            f"{config.SMOKE_RUNTIME_LIMIT_S:.0f}s. No report written."
        )
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "smoke_report.md"
    report_path.write_text(render_smoke_report(result), encoding="utf-8")
    return report_path
