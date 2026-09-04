"""Smoke run: LR + RF on the strict temporal split, gated, with report.

Trains on labeled steps 1-34, tests on labeled steps 35-49 (temporal split
only, never random; unknown-class nodes excluded from training and scoring).
Gates when BOTH models reach ROC-AUC >= 0.80 and the run stays under 10 minutes
wall clock — below/over the gate the run exits non-zero and NO report is
written (fail-closed). PR-AUC is reported from the start: it is the
predeclared challenger metric.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from aml_workbench import config, db
from aml_workbench.errors import SmokeGateError
from aml_workbench.model import fit_lr_rf, load_labeled, split_temporal
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
    runtime_limit_s: float
    passed: bool






def run_smoke(data_dir: Path) -> Path:
    """Run the smoke gate; return the report path. Fail-closed: below the
    gate (or over the runtime limit) raises SmokeGateError and writes nothing."""
    started = time.monotonic()
    x, y, steps, _ = load_labeled(db.path(data_dir))
    train_mask, test_mask = split_temporal(steps)
    metrics = [
        ModelMetrics(name=m.model, roc_auc=m.roc_auc, pr_auc=m.pr_auc)
        for m in fit_lr_rf(x, y, train_mask, test_mask, seed=config.SMOKE_SEED)
    ]
    runtime_s = time.monotonic() - started

    worst_roc = min(m.roc_auc for m in metrics)
    passed = (
        all(m.roc_auc >= config.SMOKE_ROC_AUC_GATE for m in metrics)
        and runtime_s <= config.SMOKE_RUNTIME_LIMIT_S
    )
    result = SmokeResult(
        models=metrics,
        train_base_rate=float(y[train_mask].mean()),
        test_base_rate=float(y[test_mask].mean()),
        train_rows=int(train_mask.sum()),
        test_rows=int(test_mask.sum()),
        runtime_s=runtime_s,
        gate_threshold=config.SMOKE_ROC_AUC_GATE,
        runtime_limit_s=config.SMOKE_RUNTIME_LIMIT_S,
        passed=passed,
    )
    if not passed:
        # Fail-closed: below the gate no report artifact exists.
        raise SmokeGateError(
            f"Smoke gate failed: worst ROC-AUC {worst_roc:.4f} vs threshold "
            f"{config.SMOKE_ROC_AUC_GATE} (both models must clear it), runtime "
            f"{runtime_s:.1f}s vs limit {config.SMOKE_RUNTIME_LIMIT_S:.0f}s. "
            "No report written."
        )
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "smoke_report.md"
    report_path.write_text(render_smoke_report(result), encoding="utf-8")
    return report_path
