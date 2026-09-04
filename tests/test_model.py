"""Seam tests: baselines, challenger training, predeclared PR-AUC decision,
and hyperparameter tuning. Drives the pipeline commands against a tmp_path
DuckDB seeded with a synthetic labeled Elliptic schema plus a graph-feature
model table; external behavior only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import pytest

from aml_workbench import config
from conftest import run_stages, seed_and_run_baselines, seed_workbench


def test_baselines_records_seeds_and_metrics(tmp_path: Path) -> None:
    seed_and_run_baselines(tmp_path)
    report = tmp_path / "reports" / "baselines_report.md"
    assert report.is_file()
    payload = json.loads(
        (tmp_path / "reports" / "baselines_metrics.json").read_text(encoding="utf-8")
    )
    assert len(payload["metrics"]) == 2 * len(config.MODEL_SEEDS)
    assert {m["seed"] for m in payload["metrics"]} == set(config.MODEL_SEEDS)
    for m in payload["metrics"]:
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0


def test_baselines_rf_beats_lr_on_nonlinear_fixture(tmp_path: Path) -> None:
    seed_and_run_baselines(tmp_path)
    payload = json.loads(
        (tmp_path / "reports" / "baselines_metrics.json").read_text(encoding="utf-8")
    )
    # hand expectation: f001 is radial (illicit iff |f001| > 0.8) — the
    # signal is linearly non-separable, so LR stays near base rate while RF
    # captures the band; the >= is a real comparison, not two saturated scores
    assert payload["mean_pr_auc"]["random_forest"] >= payload["mean_pr_auc"]["logistic_regression"]


def test_baselines_missing_tables_fail_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "workbench.duckdb"
    duckdb.connect(str(db_path)).close()  # empty store
    result = run_stages("baselines", data_dir=tmp_path)
    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "baselines_report.md").exists()


def test_challenger_trains_tunes_and_decides(tmp_path: Path) -> None:
    seed_and_run_baselines(tmp_path)
    result = run_stages("challenger", data_dir=tmp_path)
    assert result.exit_code == 0, result.output

    payload = json.loads(
        (tmp_path / "reports" / "challenger_metrics.json").read_text(encoding="utf-8")
    )
    assert {m["seed"] for m in payload["metrics"]} == set(config.MODEL_SEEDS)
    assert payload["best_seed"] in config.MODEL_SEEDS
    assert (tmp_path / "models" / "challenger.joblib").is_file()

    tuning = json.loads((tmp_path / "reports" / "tuning_params.json").read_text(encoding="utf-8"))
    assert set(tuning) >= set(config.LGBM_GRID)
    assert {"bagging_fraction", "bagging_freq"} <= set(tuning)  # fixed, rides with winner

    decision = (tmp_path / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert re.search(r"\*\*(promote|retain)\*\*", decision)
    assert "PR-AUC" in decision


def test_decision_retains_below_margin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_and_run_baselines(tmp_path)
    monkeypatch.setattr(config, "CHALLENGER_MIN_PR_AUC_GAIN", 2.0)
    result = run_stages("challenger", data_dir=tmp_path)
    assert result.exit_code == 0, result.output
    decision = (tmp_path / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert "**retain**" in decision


def test_decision_promotes_challenger_genuinely_wins(tmp_path: Path) -> None:
    """Real promote path: the label signal lives only in the graph features,
    so the challenger (raw + graph) beats the raw-feature baselines without any
    patched margin; the configured 0.01 minimum is met organically."""
    seed_and_run_baselines(tmp_path, graph_only=True)
    result = run_stages("challenger", data_dir=tmp_path)
    assert result.exit_code == 0, result.output
    decision = (tmp_path / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert "**promote**" in decision
    payload = json.loads(
        (tmp_path / "reports" / "challenger_metrics.json").read_text(encoding="utf-8")
    )
    baselines = json.loads(
        (tmp_path / "reports" / "baselines_metrics.json").read_text(encoding="utf-8")
    )
    assert payload["mean_pr_auc"] > max(baselines["mean_pr_auc"].values())
    # ticket: weighting + calibration + selection recorded in the artifact
    assert payload["class_weight"] == "balanced"
    assert payload["calibration"] == {"method": "isotonic", "cv": 3}
    assert "validation" in payload["seed_selection"]


def test_challenger_without_baselines_fail_closed(tmp_path: Path) -> None:
    seed_workbench(tmp_path / "workbench.duckdb")
    result = run_stages("challenger", data_dir=tmp_path)
    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "decision_report.md").exists()


def test_tuning_never_sees_test_steps(tmp_path: Path) -> None:
    """No-look-ahead proof at the tuning seam: corrupting every TEST-period
    row's features must not change the selected hyperparameters."""
    seed_and_run_baselines(tmp_path)
    assert run_stages("challenger", data_dir=tmp_path).exit_code == 0
    clean = (tmp_path / "reports" / "tuning_params.json").read_text(encoding="utf-8")

    # poison the test side: garbage features, garbage graph features
    con = duckdb.connect(str(tmp_path / "workbench.duckdb"))
    try:
        con.execute(
            "UPDATE elliptic_tx_features SET f001 = 9999.0 WHERE time_step >= $1",
            [config.TEST_STEP_MIN],
        )
        con.execute(
            "UPDATE tx_graph_features SET ego_illicit_1hop = -7.0 "
            "WHERE tx_id IN (SELECT tx_id FROM elliptic_tx_features WHERE time_step >= $1)",
            [config.TEST_STEP_MIN],
        )
    finally:
        con.close()

    assert run_stages("challenger", data_dir=tmp_path).exit_code == 0
    poisoned = (tmp_path / "reports" / "tuning_params.json").read_text(encoding="utf-8")
    assert poisoned == clean
