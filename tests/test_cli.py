"""Seam tests: the CLI is the single pipeline-command seam.

External behavior only: exit codes and listed stage names — never internals.
"""

from __future__ import annotations

from typer.testing import CliRunner

from aml_workbench.cli import app

runner = CliRunner()

ALL_STAGES = [
    "download",
    "ingest",
    "smoke",
    "rules",
    "alert-stats",
    "graph-features",
    "baselines",
    "challenger",
    "shap",
    "validate",
    "drift",
    "gnn",
    "triage",
    "view",
    "track",
    "report",
]


def _output(result) -> str:
    """CliRunner output across click versions (stderr may be separate)."""
    text = result.output
    stderr = getattr(result, "stderr", None)
    if stderr:
        text += stderr
    return text


def test_help_exits_zero_and_lists_all_stages() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for stage in ALL_STAGES:
        assert stage in _output(result), f"stage '{stage}' missing from --help"


def test_fail_closed_contract_on_missing_data(tmp_path) -> None:
    # every stage is implemented; the fail-closed contract (missing artifacts
    # -> exit 1) is covered per stage, sampled here
    result = runner.invoke(app, ["track", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
