"""Seam tests: SHAP summary on the persisted challenger model."""

from __future__ import annotations

from pathlib import Path

import duckdb
from typer.testing import CliRunner

from test_model import _run, _seed_and_run_baselines

runner = CliRunner()


def test_shap_summary_ranks_separating_feature_first(tmp_path: Path) -> None:
    _seed_and_run_baselines(tmp_path)
    assert _run("challenger", data_dir=tmp_path).exit_code == 0
    result = _run("shap", data_dir=tmp_path)
    assert result.exit_code == 0, result.output

    summary = (tmp_path / "reports" / "shap_summary.md").read_text(encoding="utf-8")
    # hand expectation: f001 = ±2 with 0.1 noise is near-perfect separation, so
    # it must carry the largest mean attribution
    assert "| 1 | f001 |" in summary
    # graph features are named in the ranking too
    assert "ego_illicit_1hop" in summary


def test_shap_without_challenger_fail_closed(tmp_path: Path) -> None:
    duckdb.connect(str(tmp_path / "workbench.duckdb")).close()
    result = _run("shap", data_dir=tmp_path)
    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "shap_summary.md").exists()
