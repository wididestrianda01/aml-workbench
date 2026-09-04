"""Seam tests for `aml track`: fail-closed on missing/empty store; manifest
contents asserted after a real tracked run (hand-written run, not internals).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app

runner = CliRunner()


def _output(result) -> str:
    text = result.output
    stderr = getattr(result, "stderr", None)
    if stderr:
        text += stderr
    return text


def test_track_fails_closed_without_store(tmp_path: Path) -> None:
    result = runner.invoke(app, ["track", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Fail-closed" in _output(result)
    assert not list((tmp_path / "reports").glob("*.json"))


def test_track_fails_closed_with_empty_store(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / config.MLFLOW_DB_NAME}")
    mlflow.set_experiment("default")
    result = runner.invoke(app, ["track", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "no runs" in _output(result)


def test_track_manifest_contents(tmp_path: Path, monkeypatch) -> None:
    """Hand-checked: one logged run with known params/metrics must appear
    verbatim in the manifest, with a commit and config fingerprint."""
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / config.MLFLOW_DB_NAME}")
    mlflow.set_experiment("hand-check")  # fresh sqlite store has no experiments yet
    with mlflow.start_run(run_name="hand-check"):
        mlflow.log_params({"model": "lgbm", "seed": 42})
        mlflow.log_metrics({"pr_auc": 0.123456789})

    result = runner.invoke(app, ["track", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, _output(result)

    manifest = json.loads((tmp_path / "reports" / "run_manifest.json").read_text())
    assert manifest["run_count"] == 1
    assert manifest["commit"] != "unknown" or os.environ.get("CI")  # repo checkout expected
    assert len(manifest["config_fingerprint"]) == 64
    run = manifest["runs"][0]
    assert run["run_name"] == "hand-check"
    assert run["params"] == {"model": "lgbm", "seed": "42"}
    assert run["metrics"]["pr_auc"] == 0.123457  # rounded to 6 dp by hand-check
    assert run["status"] == "FINISHED"
