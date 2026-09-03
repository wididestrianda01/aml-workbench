"""Seam tests: smoke run + one-page report.

Drives `aml smoke` against a tmp_path DuckDB seeded with a synthetic labeled
Elliptic schema (165 features), external behavior only. Gate violation is
injected by monkeypatching the config threshold — never by editing constants.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app

runner = CliRunner()

FEATURE_COUNT = config.FEATURE_COUNT
N_ROWS = 196  # 4 exact tiles of the 49 steps -> train 4*34=136, test 4*15=60


def _seed_smoke_db(db_path: Path, *, rng_seed: int = 0) -> None:
    """elliptic_tx + elliptic_tx_features with separable classes on both sides
    of the temporal split: f001 strongly separates illicit from licit."""
    rng = np.random.default_rng(rng_seed)
    steps = np.tile(np.arange(1, 50), 4)  # exact tiles: 136 train / 60 test rows
    labels = rng.integers(1, 3, size=N_ROWS)  # 1 illicit, 2 licit
    features = rng.normal(size=(N_ROWS, FEATURE_COUNT))
    features[:, 0] = np.where(labels == 1, 2.0, -2.0) + rng.normal(scale=0.1, size=N_ROWS)
    tx_ids = [f"{i:09d}" for i in range(N_ROWS)]

    con = duckdb.connect(str(db_path))
    try:
        cols = ", ".join(f"f{i:03d} REAL" for i in range(1, FEATURE_COUNT + 1))
        con.execute(
            "CREATE TABLE elliptic_tx (tx_id VARCHAR, class_label TINYINT)"
        )
        con.execute(
            f"CREATE TABLE elliptic_tx_features (tx_id VARCHAR, time_step SMALLINT, {cols})"
        )
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
    finally:
        con.close()


def _report(data_dir: Path) -> Path:
    return data_dir / "reports" / "smoke_report.md"


def test_smoke_green_path_writes_report(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_smoke_db(data_dir / "workbench.duckdb")
    result = runner.invoke(app, ["smoke", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    report = _report(data_dir).read_text()
    # Required content: ROC-AUC, PR-AUC, split statement, base rates, runtime, verdict.
    assert "ROC-AUC" in report and "PR-AUC" in report
    assert "1-34" in report and "35-49" in report
    assert "base rate" in report
    assert "Runtime" in report
    assert "PASS" in report
    # Gate verdict matches the threshold actually applied.
    assert f"ROC-AUC >= {config.SMOKE_ROC_AUC_GATE:.2f}" in report


def test_smoke_below_gate_fails_closed_no_report(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_smoke_db(data_dir / "workbench.duckdb")
    # Inject the violation: raise the gate above any achievable score.
    monkeypatch.setattr(config, "SMOKE_ROC_AUC_GATE", 1.01)
    result = runner.invoke(app, ["smoke", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert not _report(data_dir).exists()


def test_smoke_missing_ingest_tables_fail_closed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    duckdb.connect(str(data_dir / "workbench.duckdb")).close()
    result = runner.invoke(app, ["smoke", "--data-dir", str(data_dir)])
    assert result.exit_code == 1


def test_smoke_split_is_temporal_not_random(tmp_path: Path, monkeypatch) -> None:
    """The split boundary is a step threshold, not a random draw: steps 34 ->
    train, 35 -> test, verified through the report's row counts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_smoke_db(data_dir / "workbench.duckdb")
    result = runner.invoke(app, ["smoke", "--data-dir", str(data_dir)])
    assert result.exit_code == 0

    # 4 tiles of steps 1..49 -> train side = 4 * 34 = 136, test side = 4 * 15 = 60.
    report = _report(data_dir).read_text()
    assert "train 136 / test 60" in report
