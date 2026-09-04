"""Seam tests for `aml report`: fail-closed on missing stage artifacts; the
generated report must exist and contain the required sections.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from typer.testing import CliRunner

from aml_workbench.cli import app

runner = CliRunner()


def _output(result) -> str:
    text = result.output
    stderr = getattr(result, "stderr", None)
    if stderr:
        text += stderr
    return text


REQUIRED_SECTIONS = (
    "base rate",
    "Micro-F1",
    "Limitations",
    "Regulatory context",
    "AMLR",
    "FFFS 2017:11",
    "AI Act",
    "primer-aml-domain.md",
)


def _fake_artifacts(data_dir: Path) -> None:
    """Minimal stage artifacts: numbers are placeholders; the report's job is
    assembly, and each artifact's own seam test owns its arithmetic."""
    reports = data_dir / "reports"
    reports.mkdir(parents=True)
    payloads = {
        "run_manifest.json": {
            "commit": "test", "config_fingerprint": "x" * 64, "run_count": 1, "runs": [],
        },
        "baselines_metrics.json": {
            "seeds": [42],
            "metrics": [
                {"model": "logistic_regression", "seed": 42, "roc_auc": 0.5, "pr_auc": 0.1},
                {"model": "random_forest", "seed": 42, "roc_auc": 0.6, "pr_auc": 0.2},
            ],
        },
        "challenger_metrics.json": {
            "seeds": [42], "best_seed": 42, "mean_pr_auc": 0.3,
            "metrics": [
                {"model": "lightgbm_challenger", "seed": 42, "roc_auc": 0.7, "pr_auc": 0.3},
            ],
        },
        "tuning_params.json": {"num_leaves": 31},
        "validation_metrics.json": {
            "per_step": [
                {"step": 35, "n_eval": 10, "illicit_rate": 0.1, "f1": 0.5,
                 "precision": 0.5, "recall": 0.5, "pr_auc": 0.4},
            ],
        },
        "drift_metrics.json": {
            "columns": {
                "f001": {"psi": 0.05, "breach": False, "watch": False},
                "f002": {"psi": 0.3, "breach": True, "watch": True},
            },
        },
        "gnn_comparison.json": {
            "gnn": {"mean_roc_auc": 0.5, "mean_pr_auc": 0.1, "std_roc_auc": 0.0,
                    "std_pr_auc": 0.0, "seeds": [42]},
            "challenger": {"mean_roc_auc": 0.7, "mean_pr_auc": 0.3, "std_roc_auc": 0.0,
                           "std_pr_auc": 0.0, "seeds": [42]},
            "pr_auc_delta": 0.2, "verdict": "GNN loses to the GBM",
        },
    }
    for name, payload in payloads.items():
        (reports / name).write_text(json.dumps(payload))


def _fake_db(data_dir: Path) -> None:
    con = duckdb.connect(str(data_dir / "workbench.duckdb"))
    con.execute(
        "CREATE TABLE elliptic_tx_features AS"
        " SELECT 1::SMALLINT AS time_step, 'a'::VARCHAR AS tx_id"
    )
    con.execute(
        "CREATE TABLE elliptic_tx AS SELECT 'a'::VARCHAR AS tx_id, '1'::VARCHAR AS class_label"
    )
    con.execute(
        "CREATE TABLE alert_queue AS SELECT 1::BIGINT AS rank, 'fan-in'::VARCHAR AS scenario,"
        " 'e'::VARCHAR AS entity, 0.5::DOUBLE AS fused_score"
    )
    con.execute(
        "CREATE TABLE triage_meta AS SELECT DATE '2022-09-11' AS day1,"
        " DATE '2022-09-15' AS day2, 1::INTEGER AS universe, 1::INTEGER AS days_eval"
    )
    con.execute(
        "CREATE TABLE triage_kpi AS SELECT 'overall'::VARCHAR AS scope,"
        " 1::BIGINT AS true_positives, 0.1::DOUBLE AS detection_rate,"
        " 0.5::DOUBLE AS precision_at_100, 0.4::DOUBLE AS precision_at_500,"
        " 0.3::DOUBLE AS precision_at_1000"
    )
    con.execute(
        "CREATE TABLE hismall_transaction AS SELECT TIMESTAMP '2022-09-20' AS tx_time,"
        " 0::UTINYINT AS is_laundering, 'a'::VARCHAR AS to_account, 'b'::VARCHAR AS from_account"
    )
    con.close()


def test_report_fails_closed_on_missing_artifact(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing" in _output(result)

def test_report_writes_document_with_required_sections(tmp_path: Path, monkeypatch) -> None:
    _fake_artifacts(tmp_path)
    _fake_db(tmp_path)
    from aml_workbench import config as cfg
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)  # docs land in tmp, not the repo
    result = runner.invoke(app, ["report", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, _output(result)

    report = (tmp_path / "docs" / "report.md").read_text()
    for section in REQUIRED_SECTIONS:
        assert section in report, f"required section '{section}' missing"
    assert (tmp_path / "docs" / "figures" / "base_rate_curve.png").stat().st_size > 0
    assert (tmp_path / "docs" / "figures" / "precision_at_k.png").stat().st_size > 0
