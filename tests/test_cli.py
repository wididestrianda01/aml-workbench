"""P1-01 seam tests: the CLI is the single pipeline-command seam.

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


def test_unimplemented_stage_fails_closed() -> None:
    for stage, phase in [("rules", 2), ("graph-features", 3), ("report", 8)]:
        result = runner.invoke(app, [stage])
        assert result.exit_code == 1, f"{stage} must exit non-zero"
        assert "not implemented" in _output(result)
        assert f"Phase {phase}" in _output(result)
