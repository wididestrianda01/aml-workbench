"""P2 seam tests: per-scenario alert statistics (`aml alert-stats`).

External behavior only: exit codes, stats-table contents, report artifact.
"""

from __future__ import annotations

import duckdb
from typer.testing import CliRunner

from aml_workbench.cli import app
from conftest import make_hismall_data_dir, run_ingest
from test_rules import _accounts, _ingest_and_run_rules, _patch_gates

runner = CliRunner()


def test_alert_stats_after_rules(tmp_path, monkeypatch) -> None:
    # One structuring campaign (A->B, 3 x 9500..9900 same day, total 29100).
    tx_rows = [
        ["2022/09/01 03:00", "010", "A", "010", "B",
         "9500.00", "US Dollar", "9500.00", "US Dollar", "ACH", "0"],
        ["2022/09/01 05:00", "010", "A", "010", "B",
         "9700.00", "US Dollar", "9700.00", "US Dollar", "ACH", "0"],
        ["2022/09/01 07:00", "010", "A", "010", "B",
         "9900.00", "US Dollar", "9900.00", "US Dollar", "ACH", "0"],
        ["2022/09/02 08:00", "010", "A", "010", "B",
         "9900.00", "US Dollar", "9900.00", "US Dollar", "ACH", "1"],
    ]
    data_dir = _ingest_and_run_rules(
        monkeypatch, tmp_path, tx_rows, ["A", "B"]
    )
    result = runner.invoke(app, ["alert-stats", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output

    report = data_dir / "reports" / "alert_stats.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "structuring" in text and "Alerts" in text

    con = duckdb.connect(str(data_dir / "workbench.duckdb"), read_only=True)
    try:
        stats = con.execute(
            "SELECT scenario, alert_count, distinct_entities, "
            "laundering_entity_precision FROM alert_scenario_stats "
            "ORDER BY scenario"
        ).fetchall()
    finally:
        con.close()
    # 1 structuring alert: the GROUP BY collapses the 3-tx same-day campaign
    # into one row (the 09/02 tx is single and in-band but alone in its day).
    # Entity A is laundering-flagged (the 09/02 row), so precision = 1.0.
    assert stats == [("structuring", 1, 1, 1.0)]


def test_alert_stats_fails_closed_without_rules(tmp_path, monkeypatch) -> None:
    tx_rows = [
        ["2022/09/01 03:00", "010", "A", "010", "B",
         "100.00", "US Dollar", "100.00", "US Dollar", "ACH", "0"],
    ]
    data_dir = make_hismall_data_dir(tmp_path, tx_rows, _accounts("A", "B"))
    _patch_gates(monkeypatch, tx_rows, ["A", "B"])
    assert run_ingest(data_dir, "hi-small").exit_code == 0

    result = runner.invoke(app, ["alert-stats", "--data-dir", str(data_dir)])
    assert "rule_alert" in result.output
    assert not (data_dir / "reports" / "alert_stats.md").exists()


def test_alert_stats_fails_closed_on_empty_alert_table(tmp_path, monkeypatch) -> None:
    # An alert table with zero rows means every scenario fired on nothing —
    # that is a broken run, not an empty result, so alert-stats stops.
    tx_rows = [
        ["2022/09/01 03:00", "010", "A", "010", "B",
         "100.00", "US Dollar", "100.00", "US Dollar", "ACH", "0"],
    ]
    data_dir = make_hismall_data_dir(tmp_path, tx_rows, _accounts("A", "B"))
    _patch_gates(monkeypatch, tx_rows, ["A", "B"])
    assert run_ingest(data_dir, "hi-small").exit_code == 0

    con = duckdb.connect(str(data_dir / "workbench.duckdb"))
    try:
        con.execute(
            "CREATE OR REPLACE TABLE rule_alert AS "
            "SELECT 'structuring' AS scenario, 'A' AS entity, "
            "'x' AS reason, 'y' AS details WHERE false"
        )
    finally:
        con.close()

    result = runner.invoke(app, ["alert-stats", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "empty" in result.output
    assert not (data_dir / "reports" / "alert_stats.md").exists()
