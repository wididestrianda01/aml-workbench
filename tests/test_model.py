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
import numpy as np
import pytest
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app

runner = CliRunner()

FEATURE_COUNT = config.FEATURE_COUNT
N_ROWS = 196  # 4 exact tiles of the 49 steps -> train 136 / test 60


def _seed_db(db_path: Path, *, rng_seed: int = 0) -> None:
    """elliptic_tx + elliptic_tx_features + tx_graph_features. f001 separates
    the classes on both sides of the split; ego_illicit_1hop mirrors the label
    so the graph features carry signal too."""
    rng = np.random.default_rng(rng_seed)
    steps = np.tile(np.arange(1, 50), 4)
    labels = rng.integers(1, 3, size=N_ROWS)  # 1 illicit, 2 licit
    features = rng.normal(size=(N_ROWS, FEATURE_COUNT))
    features[:, 0] = np.where(labels == 1, 2.0, -2.0) + rng.normal(scale=0.1, size=N_ROWS)
    tx_ids = [f"{i:09d}" for i in range(N_ROWS)]

    con = duckdb.connect(str(db_path))
    try:
        cols = ", ".join(f"f{i:03d} REAL" for i in range(1, FEATURE_COUNT + 1))
        con.execute("CREATE TABLE elliptic_tx (tx_id VARCHAR, class_label TINYINT)")
        con.execute(
            f"CREATE TABLE elliptic_tx_features (tx_id VARCHAR, time_step SMALLINT, {cols})"
        )
        graph_cols = (
            "in_degree INTEGER, out_degree INTEGER, reciprocity DOUBLE, "
            "ego_illicit_1hop DOUBLE, ego_illicit_2hop DOUBLE, "
            "louvain_community INTEGER, time_since_activity SMALLINT"
        )
        con.execute(f"CREATE TABLE tx_graph_features (tx_id VARCHAR, {graph_cols})")
        con.executemany(
            "INSERT INTO elliptic_tx VALUES (?, ?)",
            [(tx_ids[i], int(labels[i])) for i in range(N_ROWS)],
        )
        con.executemany(
            f"INSERT INTO elliptic_tx_features VALUES (?, ?, {', '.join(['?'] * FEATURE_COUNT)})",
            [
                (tx_ids[i], int(steps[i]), *[float(v) for v in features[i]])
                for i in range(N_ROWS)
            ],
        )
        con.executemany(
            "INSERT INTO tx_graph_features VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    tx_ids[i],
                    int(rng.integers(0, 6)),
                    int(rng.integers(0, 6)),
                    float(rng.integers(0, 2)),
                    float(labels[i] == 1) + float(rng.normal(scale=0.05)),
                    float(labels[i] == 1) + float(rng.normal(scale=0.1)),
                    int(rng.integers(0, 4)),
                    int(rng.integers(0, 3)),
                )
                for i in range(N_ROWS)
            ],
        )
    finally:
        con.close()


def _run(*args: str, data_dir: Path) -> object:
    return runner.invoke(app, [*args, "--data-dir", str(data_dir)])


def _seed_and_run_baselines(tmp_path: Path) -> None:
    _seed_db(tmp_path / "workbench.duckdb")
    result = _run("baselines", data_dir=tmp_path)
    assert result.exit_code == 0, result.output


def test_baselines_records_seeds_and_metrics(tmp_path: Path) -> None:
    _seed_and_run_baselines(tmp_path)
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


def test_baselines_rf_beats_lr_on_separable_fixture(tmp_path: Path) -> None:
    _seed_and_run_baselines(tmp_path)
    payload = json.loads(
        (tmp_path / "reports" / "baselines_metrics.json").read_text(encoding="utf-8")
    )
    # hand expectation: f001 = ±2 with 0.1 noise separates classes almost
    # perfectly; both models saturate, RF never worse than LR on this fixture
    assert (
        payload["mean_pr_auc"]["random_forest"]
        >= payload["mean_pr_auc"]["logistic_regression"]
    )


def test_baselines_missing_tables_fail_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "workbench.duckdb"
    duckdb.connect(str(db_path)).close()  # empty store
    result = _run("baselines", data_dir=tmp_path)
    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "baselines_report.md").exists()


def test_challenger_trains_tunes_and_decides(tmp_path: Path) -> None:
    _seed_and_run_baselines(tmp_path)
    result = _run("challenger", data_dir=tmp_path)
    assert result.exit_code == 0, result.output

    payload = json.loads(
        (tmp_path / "reports" / "challenger_metrics.json").read_text(encoding="utf-8")
    )
    assert {m["seed"] for m in payload["metrics"]} == set(config.MODEL_SEEDS)
    assert payload["best_seed"] in config.MODEL_SEEDS
    assert (tmp_path / "models" / "challenger.joblib").is_file()

    tuning = json.loads(
        (tmp_path / "reports" / "tuning_params.json").read_text(encoding="utf-8")
    )
    assert set(tuning) == set(config.LGBM_GRID)

    decision = (tmp_path / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert re.search(r"\*\*(promote|retain)\*\*", decision)
    assert "PR-AUC" in decision


def test_decision_retains_below_margin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_and_run_baselines(tmp_path)
    monkeypatch.setattr(config, "CHALLENGER_MIN_PR_AUC_GAIN", 2.0)
    result = _run("challenger", data_dir=tmp_path)
    assert result.exit_code == 0, result.output
    decision = (tmp_path / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert "**retain**" in decision


def test_decision_promotes_above_zero_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_and_run_baselines(tmp_path)
    monkeypatch.setattr(config, "CHALLENGER_MIN_PR_AUC_GAIN", -1.0)
    result = _run("challenger", data_dir=tmp_path)
    assert result.exit_code == 0, result.output
    decision = (tmp_path / "reports" / "decision_report.md").read_text(encoding="utf-8")
    assert "**promote**" in decision


def test_challenger_without_baselines_fail_closed(tmp_path: Path) -> None:
    _seed_db(tmp_path / "workbench.duckdb")
    result = _run("challenger", data_dir=tmp_path)
    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "decision_report.md").exists()


def test_tuning_never_sees_test_steps(tmp_path: Path) -> None:
    """No-look-ahead proof at the tuning seam: corrupting every TEST-period
    row's features must not change the selected hyperparameters."""
    _seed_and_run_baselines(tmp_path)
    assert _run("challenger", data_dir=tmp_path).exit_code == 0
    clean = (tmp_path / "reports" / "tuning_params.json").read_text(encoding="utf-8")

    # poison the test side: garbage features, garbage graph features
    con = duckdb.connect(str(tmp_path / "workbench.duckdb"))
    try:
        con.execute(
            "UPDATE elliptic_tx_features SET f001 = 9999.0 "
            "WHERE time_step >= $1",
            [config.TEST_STEP_MIN],
        )
        con.execute(
            "UPDATE tx_graph_features SET ego_illicit_1hop = -7.0 "
            "WHERE tx_id IN (SELECT tx_id FROM elliptic_tx_features WHERE time_step >= $1)",
            [config.TEST_STEP_MIN],
        )
    finally:
        con.close()

    assert _run("challenger", data_dir=tmp_path).exit_code == 0
    poisoned = (tmp_path / "reports" / "tuning_params.json").read_text(encoding="utf-8")
    assert poisoned == clean
