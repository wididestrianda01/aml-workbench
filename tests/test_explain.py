"""Seam tests: SHAP summary on the persisted challenger model."""

from __future__ import annotations

from pathlib import Path

import duckdb
from typer.testing import CliRunner

from conftest import run_stages, seed_and_run_baselines

runner = CliRunner()


def test_shap_summary_ranks_separating_feature_first(tmp_path: Path) -> None:
    seed_and_run_baselines(tmp_path)
    assert run_stages("challenger", data_dir=tmp_path).exit_code == 0
    result = run_stages("shap", data_dir=tmp_path)
    assert result.exit_code == 0, result.output

    summary = (tmp_path / "reports" / "shap_summary.md").read_text(encoding="utf-8")
    # hand expectation: f001 carries the radial band signal (illicit iff
    # |f001| > 0.8), so it must rank first by mean absolute attribution
    assert "| 1 | f001 |" in summary
    # graph features are named in the ranking too
    assert "ego_illicit_1hop" in summary


def test_shap_without_challenger_fail_closed(tmp_path: Path) -> None:
    duckdb.connect(str(tmp_path / "workbench.duckdb")).close()
    result = run_stages("shap", data_dir=tmp_path)
    assert result.exit_code == 1
    assert not (tmp_path / "reports" / "shap_summary.md").exists()
