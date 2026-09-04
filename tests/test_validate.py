"""Seam tests for the validation stage: run `aml validate` against the seeded
store and assert external behavior only — exit codes, artifact contents,
split-boundary enforcement, no-look-ahead ordering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aml_workbench.errors import DataQualityError
from conftest import run_stages, seed_workbench


def _seed(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    seed_workbench(data_dir / "workbench.duckdb")
    return data_dir


def test_validate_runs_and_writes_artifact(tmp_path: Path) -> None:
    data_dir = _seed(tmp_path)
    result = run_stages("validate", data_dir=data_dir)
    assert result.exit_code == 0, result.output
    metrics = json.loads((data_dir / "reports" / "validation_metrics.json").read_text())
    # per-timestep artifact covers every test step 35-49
    per_step = {m["step"]: m for m in metrics["per_step"]}
    assert set(per_step) == set(range(35, 50))
    for m in per_step.values():
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0
        assert m["n_eval"] == 4  # fixture: 4 tiles per step
    # base-rate curve covers all 49 steps
    curve = {c["step"]: c["illicit_rate"] for c in metrics["base_rate_curve"]}
    assert set(curve) == set(range(1, 50))
    # micro-F1 caveat is written into the output
    report = (data_dir / "reports" / "validation_report.md").read_text()
    assert "micro-F1" in report and "majority" in report


def test_validate_fail_closed_on_missing_tables(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = run_stages("validate", data_dir=data_dir)
    assert result.exit_code != 0
    assert "Fail-closed" in result.output


def test_train_mask_is_strictly_before_as_of() -> None:
    """Sampled as-of steps: every mask row must come strictly before the
    as-of step — the no-look-ahead contract on the split itself."""
    from aml_workbench.validate import train_mask_before

    steps = np.tile(np.arange(1, 50), 4)
    for as_of in (35, 38, 42, 46, 49):  # sampled, incl. the split boundary
        mask = train_mask_before(steps, as_of)
        assert mask.sum() > 0
        assert steps[mask].max() < as_of


def test_leaky_split_is_rejected_fail_closed(tmp_path: Path) -> None:
    """Red-then-green: a deliberately leaky split (training rows at or after
    the as-of step) must be caught by the boundary assertion and fail closed,
    not silently produce metrics."""

    def leaky_mask(steps: np.ndarray, as_of: int) -> np.ndarray:
        # includes the as-of step itself — the look-ahead a leaky
        # implementation lets through
        return steps <= as_of

    data_dir = _seed(tmp_path)
    with pytest.raises(DataQualityError):
        from aml_workbench.validate import run_validation

        run_validation(data_dir, train_mask_fn=leaky_mask)
