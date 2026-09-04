"""Triage seam tests: fused queue ranking, hand-checked KPI arithmetic, and
the strict-inductive scoring guarantee (features strictly before the alert's
own day), exercised through the `aml triage` pipeline command."""

from __future__ import annotations

from pathlib import Path

import duckdb
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app
from conftest import make_hismall_data_dir, run_ingest

runner = CliRunner()


def _rows(*rows: str) -> list[list[str]]:
    return [row.split(",") for row in rows]


def _accounts(*accounts: str) -> list[list[str]]:
    return [[f"Bank {a}", "001", a, f"E{a}", f"Entity {a}"] for a in accounts]


# --- seeded-fixture helpers ------------------------------------------------------


def _tx_rows() -> list[list[str]]:
    """Twelve-day timeline: known account behavior, one laundering payment per
    key account, two fan-out alerts and one structuring alert."""
    rows: list[list[str]] = [
        "2022/09/03 10:00,010,A1,010,P,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/05 10:00,010,A2,010,Q,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/09 10:00,010,SRC,010,P,5000.00,US Dollar,5000.00,US Dollar,Wire,1",
        "2022/09/11 10:00,010,SRC2,010,Z,5000.00,US Dollar,5000.00,US Dollar,Wire,1",
        "2022/09/11 11:00,010,SRC3,010,Q,5000.00,US Dollar,5000.00,US Dollar,Wire,1",
        "2022/09/09 12:00,010,X,010,D,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
        "2022/09/09 13:00,010,X,010,D,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
        "2022/09/09 14:00,010,X,010,D,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
    ]
    for day, sender, receivers in (
        ("2022/09/09", "S", ["R1", "R2", "R3", "R4", "R5"]),
        ("2022/09/11", "B", ["C1", "C2", "C3", "C4", "C5"]),
    ):
        for i, r in enumerate(receivers):
            rows.append(
                f"{day} 1{i}:00,010,{sender},010,{r},20000.00,US Dollar,"
                f"20000.00,US Dollar,Wire,0"
            )
    return [row.split(",") for row in rows]


def _seed_triage_db(data_dir: Path) -> None:
    """Build workbench.duckdb directly: hismall_transaction + a seeded
    rule_alert/alert_scenario_stats pair, so queue and KPI expectations are
    exactly hand-computable."""
    con = duckdb.connect(str(data_dir / "workbench.duckdb"))
    tx = _tx_rows()
    values = ",".join(
        f"('{r[0]}','{r[1]}','{r[2]}','{r[3]}','{r[4]}',{r[5]},'{r[6]}',"
        f"{r[7]},'{r[8]}','{r[9]}',{r[10]})"
        for r in tx
    )
    con.execute(
        """
        CREATE TABLE hismall_transaction AS
        SELECT strptime(ts, '%Y/%m/%d %H:%M') AS tx_time,
               from_bank, from_account, to_bank, to_account,
               amount_received, receiving_currency, amount_paid,
               payment_currency, payment_format, is_laundering
        FROM (VALUES
        """ + values + """
        ) AS t(ts, from_bank, from_account, to_bank, to_account,
               amount_received, receiving_currency, amount_paid,
               payment_currency, payment_format, is_laundering)
        """
    )
    con.execute(
        """
        CREATE TABLE rule_alert AS SELECT * FROM (VALUES
            ('fan-out', 'S', 'concentration', 'total_usd=100000.00;window=2022-09-09'),
            ('fan-out', 'Q', 'concentration', 'total_usd=100000.00;window=2022-09-11'),
            ('fan-out', 'B', 'concentration', 'total_usd=100000.00;window=2022-09-11'),
            ('structuring', 'X', 'band', 'total_usd=28500.00;window=2022-09-09')
        ) AS t(scenario, entity, reason, details)
        """
    )
    con.execute(
        """
        CREATE TABLE alert_scenario_stats AS SELECT * FROM (VALUES
            ('fan-out', 3, 3, 0.5),
            ('structuring', 1, 1, 0.0)
        ) AS t(scenario, alert_count, distinct_entities, laundering_entity_precision)
        """
    )
    con.close()


def _run_triage(monkeypatch, tmp_path: Path, data_dir: Path):
    monkeypatch.setattr(config, "TRIAGE_OPERATING_K", 100)
    monkeypatch.setattr(config, "PRECISION_AT_K", (4,))
    monkeypatch.setattr(config, "INVESTIGATION_COST_USD", 50.0)
    return runner.invoke(app, ["triage", "--data-dir", str(data_dir)])


# --- fail-closed -----------------------------------------------------------------


def test_triage_fails_closed_without_database(tmp_path) -> None:
    result = runner.invoke(app, ["triage", "--data-dir", str(tmp_path / "empty")])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_triage_fails_closed_on_empty_alerts(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    con = duckdb.connect(str(data_dir / "workbench.duckdb"))
    con.execute("CREATE TABLE rule_alert AS SELECT 's' AS scenario, 'e' AS entity, "
                "'r' AS reason, 'd' AS details WHERE false")
    con.execute("CREATE TABLE alert_scenario_stats AS SELECT 's' AS scenario, "
                "1 AS alert_count, 1 AS distinct_entities, 0.0 AS laundering_entity_precision "
                "WHERE false")
    con.execute("CREATE TABLE hismall_transaction AS SELECT * FROM rule_alert WHERE false")
    con.close()
    result = runner.invoke(app, ["triage", "--data-dir", str(data_dir)])
    assert result.exit_code != 0
    assert "empty" in result.output


# --- full queue + KPI hand-check --------------------------------------------------


def test_triage_queue_ranking_and_kpi_arithmetic(tmp_path, monkeypatch) -> None:
    # Hand-computed on the seeded fixture (cut points on the 09/03-09/11 span:
    # day1 = 09/08, day2 = 09/09; evaluation window (09/09, 09/11] = 2 days):
    # - laundering accounts in eval window: {Q, Z, SRC2, SRC3} (senders count
    #   too) -> 22 accounts overall; FPR = 3 / (22 - 4) = 0.166667
    # - overall at k >= 4: detection 1/4, precision@4 = 1/4, yield 1/4,
    #   alerts/day 4/2 = 2, cost per TP = 4 * 50 / 1 = 200
    # - fan-out scope: 3 alerts, 1 TP -> detection 0.25, FPR 0.111111,
    #   yield 1/3, cost per TP 150
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_triage_db(data_dir)
    result = _run_triage(monkeypatch, tmp_path, data_dir)
    assert result.exit_code == 0, result.output

    con = duckdb.connect(str(data_dir / "workbench.duckdb"), read_only=True)
    queue = con.execute("SELECT * FROM alert_queue").fetchdf()
    kpi = con.execute("SELECT * FROM triage_kpi").fetchdf()
    meta = con.execute("SELECT day1, day2, universe, days_eval FROM triage_meta").fetchone()
    con.close()

    # ranking invariant: rank 1..n strictly follows fused score (desc)
    assert list(queue["rank"]) == list(range(1, len(queue) + 1))
    assert queue["fused_score"].is_monotonic_decreasing
    assert len(queue) == 4

    # strict inductive: no-prior-history alerts carry no ML component and fuse
    # to the rule component alone
    no_history = queue[queue["entity"].isin(["S", "B"])]
    assert no_history["ml_component"].isna().all()
    assert (no_history["fused_score"] == no_history["rule_component"]).all()

    overall = kpi[kpi["scope"] == "overall"].iloc[0]
    assert overall["alerts_investigated"] == 4
    assert overall["true_positives"] == 1
    assert overall["detection_rate"] == 0.25
    assert overall["false_positive_rate"] == 0.166667
    assert overall["alerts_per_day"] == 2.0
    assert overall["investigation_yield"] == 0.25
    assert overall["precision_at_4"] == 0.25
    assert overall["cost_per_true_positive"] == 200.0

    fanout = kpi[kpi["scope"] == "fan-out"].iloc[0]
    assert fanout["alerts_investigated"] == 3
    assert fanout["true_positives"] == 1
    assert fanout["detection_rate"] == 0.25
    assert fanout["false_positive_rate"] == 0.111111
    assert fanout["investigation_yield"] == 0.3333
    assert fanout["precision_at_4"] == 0.3333
    assert fanout["cost_per_true_positive"] == 150.0

    # meta table records the walk-forward cut points for the view
    assert meta[2] == 22 and meta[3] == 2

    report = data_dir / "reports" / "triage.md"
    assert report.exists()
    assert "Operational KPIs" in report.read_text()


# --- full seam: ingest -> rules -> alert-stats -> triage --------------------------


def test_triage_runs_end_to_end_after_rules(tmp_path, monkeypatch) -> None:
    import test_rules

    tx_rows = _tx_rows()
    accounts = sorted({r[2] for r in tx_rows} | {r[4] for r in tx_rows})
    data_dir = make_hismall_data_dir(tmp_path, tx_rows, _accounts(*accounts))
    test_rules._patch_gates(monkeypatch, tx_rows, accounts)
    assert run_ingest(data_dir, "hi-small").exit_code == 0
    assert runner.invoke(app, ["rules", "--data-dir", str(data_dir)]).exit_code == 0
    assert runner.invoke(app, ["alert-stats", "--data-dir", str(data_dir)]).exit_code == 0
    monkeypatch.setattr(config, "TRIAGE_OPERATING_K", 10)
    monkeypatch.setattr(config, "PRECISION_AT_K", (1, 2))
    result = runner.invoke(app, ["triage", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output

    con = duckdb.connect(str(data_dir / "workbench.duckdb"), read_only=True)
    queue = con.execute("SELECT * FROM alert_queue").fetchdf()
    con.close()
    assert len(queue) > 0
    assert queue["fused_score"].is_monotonic_decreasing
    # the real fan-out typology (S, B) must be present in the queue
    assert set(queue["scenario"]) >= {"fan-out"}
    assert (data_dir / "reports" / "triage.md").exists()
