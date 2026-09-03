"""P2 seam tests: rules-based scenario engine over HI-Small tables.

Every test drives the pipeline seam (`aml ingest` -> `aml rules`) against
constructed typology fixtures with hand-computed expectations written inline
BEFORE the assertions run. Each rule must fire exactly on its typology and
stay silent on benign look-alikes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app
from conftest import make_hismall_data_dir, run_ingest

runner = CliRunner()


def _rows(*rows: list[str]) -> list[list[str]]:
    return [row.split(",") for row in rows]


def _accounts(*accounts: str) -> list[list[str]]:
    """One account row per distinct account used by the scenario fixtures."""
    return [[f"Bank {a}", "001", a, f"E{a}", f"Entity {a}"] for a in accounts]


def _patch_gates(monkeypatch, tx_rows: list[list[str]], accounts: list[str]) -> None:
    laundering = sum(1 for row in tx_rows if row[-1] == "1")
    monkeypatch.setattr(config, "HI_SMALL_MIN_TX", len(tx_rows))
    monkeypatch.setattr(config, "HI_SMALL_MIN_ACCOUNTS", len(accounts))
    monkeypatch.setattr(config, "HI_SMALL_LAUNDERING_COUNT_PINNED", laundering)
    monkeypatch.setattr(
        config, "HI_SMALL_LAUNDERING_RATE_TARGET", laundering / len(tx_rows)
    )


def _ingest_and_run_rules(
    monkeypatch, tmp_path: Path, tx_rows: list[list[str]], accounts: list[str]
) -> Path:
    data_dir = make_hismall_data_dir(tmp_path, tx_rows, _accounts(*accounts))
    _patch_gates(monkeypatch, tx_rows, accounts)
    assert run_ingest(data_dir, "hi-small").exit_code == 0
    result = runner.invoke(app, ["rules", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    return data_dir


def _alerts(data_dir: Path) -> list[tuple]:
    con = duckdb.connect(str(data_dir / "workbench.duckdb"), read_only=True)
    try:
        return con.execute(
            "SELECT scenario, entity, details FROM rule_alert "
            "ORDER BY scenario, entity, details"
        ).fetchall()
    finally:
        con.close()


# --- structuring------------------------------------------------------


def test_structuring_fires_and_benign_look_alikes_stay_silent(
    tmp_path, monkeypatch
) -> None:
    # Hand-computed: A->B sends 3 payments in the [9000, 10000) band on one day,
    # total 9500 + 9700 + 9900 = 29100 >= 10000 -> 1 structuring alert (entity A).
    # Benign: C->D one sub-threshold payment (1 < 3); E->F three payments of 50
    # (below band floor); G->H three in-band payments spread over 3 days
    # (tumbling daily window -> never 3 in one day).
    tx_rows = _rows(
        "2022/09/01 00:20,010,L1,010,L2,100000.00,US Dollar,100000.00,US Dollar,ACH,1",
        "2022/09/01 03:00,010,A,010,B,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
        "2022/09/01 05:00,010,A,010,B,9700.00,US Dollar,9700.00,US Dollar,ACH,0",
        "2022/09/01 07:00,010,A,010,B,9900.00,US Dollar,9900.00,US Dollar,ACH,0",
        "2022/09/01 10:00,010,C,010,D,9900.00,US Dollar,9900.00,US Dollar,ACH,0",
        "2022/09/01 11:00,010,E,010,F,50.00,US Dollar,50.00,US Dollar,ACH,0",
        "2022/09/01 12:00,010,E,010,F,50.00,US Dollar,50.00,US Dollar,ACH,0",
        "2022/09/01 13:00,010,E,010,F,50.00,US Dollar,50.00,US Dollar,ACH,0",
        "2022/09/01 14:00,010,G,010,H,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
        "2022/09/02 14:00,010,G,010,H,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
        "2022/09/03 14:00,010,G,010,H,9500.00,US Dollar,9500.00,US Dollar,ACH,0",
    )
    accounts = ["A", "B", "C", "D", "E", "F", "G", "H", "L1", "L2"]
    data_dir = _ingest_and_run_rules(monkeypatch, tmp_path, tx_rows, accounts)

    alerts = _alerts(data_dir)
    assert [(s, e) for s, e, _ in alerts] == [("structuring", "A")]
    assert "txs=3" in alerts[0][2]
    assert "counterparties=1" in alerts[0][2]


def test_rules_fails_closed_without_database(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "empty"
    data_dir.mkdir()
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)
    result = runner.invoke(app, ["rules", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "not found" in result.output


# --- velocity + rapid churn-------------------------------------------


def test_velocity_fires_on_burst_and_benign_steady_stays_silent(
    tmp_path, monkeypatch
) -> None:
    # Hand-computed: V sends exactly 20 outgoing txs in one day (threshold is
    # >= 20, so this is the boundary) -> 1 velocity alert (entity V).
    # Benign: W sends 3 txs the same day -> below threshold, silent.
    rows = [
        f"2022/09/01 {h % 12 + 8:02d}:{h % 60:02d},010,V,010,V{h + 1},"
        "100.00,US Dollar,100.00,US Dollar,ACH,0"
        for h in range(20)  # V -> V1..V20, 20 txs, 100 each (total 2000 < fan-out)
    ] + [
        "2022/09/01 09:00,010,W,010,W1,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/01 10:00,010,W,010,W2,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/01 11:00,010,W,010,W3,100.00,US Dollar,100.00,US Dollar,ACH,0",
        # laundering-flagged row, isolated, fires nothing (1 counterparty, 1 tx)
        "2022/09/02 00:20,010,L1,010,L2,100000.00,US Dollar,100000.00,US Dollar,Wire,1",
    ]
    tx_rows = _rows(*rows)
    accounts = ["V", *(f"V{h}" for h in range(1, 21)), "W", "W1", "W2", "W3", "L1", "L2"]
    data_dir = _ingest_and_run_rules(monkeypatch, tmp_path, tx_rows, accounts)

    alerts = _alerts(data_dir)
    assert [(s, e) for s, e, _ in alerts] == [("velocity", "V")]
    assert "txs=20" in alerts[0][2]


def test_rapid_churn_fires_on_passthrough_and_holders_stay_silent(
    tmp_path, monkeypatch
) -> None:
    # Hand-computed: X -> P 10000.00 at 09:00; P -> Y 9800.00 at 11:00.
    # Payout 9800 >= (1 - 0.10) * 10000 = 9000 -> churn fires (entity P,
    # retained 2.0%). Benign: Z -> Q 10000.00 at 09:00; Q -> R 2000.00 at
    # 11:00 -> payout 2000 < 9000 (80% retained), silent.
    tx_rows = _rows(
        "2022/09/01 00:20,010,L1,010,L2,100000.00,US Dollar,100000.00,US Dollar,Wire,1",
        "2022/09/01 09:00,010,X,010,P,10000.00,US Dollar,10000.00,US Dollar,Wire,0",
        "2022/09/01 11:00,010,P,010,Y,9800.00,US Dollar,9800.00,US Dollar,Wire,0",
        "2022/09/01 09:00,010,Z,010,Q,10000.00,US Dollar,10000.00,US Dollar,Wire,0",
        "2022/09/01 11:00,010,Q,010,R,2000.00,US Dollar,2000.00,US Dollar,Wire,0",
    )
    accounts = ["L1", "L2", "X", "P", "Y", "Z", "Q", "R"]
    data_dir = _ingest_and_run_rules(monkeypatch, tmp_path, tx_rows, accounts)

    alerts = _alerts(data_dir)
    assert [(s, e) for s, e, _ in alerts] == [("rapid-churn", "P")]
    assert "received_usd=10000.0" in alerts[0][2]
    assert "retained_pct=2.0" in alerts[0][2]


# --- fan-in / fan-out-------------------------------------------------


def test_fan_in_fan_out_fire_on_concentration_and_below_threshold_stays_silent(
    tmp_path, monkeypatch
) -> None:
    # Hand-computed (thresholds: >= 5 distinct counterparties AND >= 50000 USD
    # aggregate in one day):
    # - F1..F5 -> T, 20000 each: 5 counterparties, 100000 total -> fan-in (T).
    # - G1..G4 -> U, 20000 each: 4 counterparties -> below threshold, silent
    #   (boundary: exactly 4 does not fire).
    # - S -> R1..R5, 20000 each: 5 counterparties, 100000 total -> fan-out (S).
    # - S2 -> Q1..Q4, 20000 each: 4 counterparties, silent.
    senders_in = ["F1", "F2", "F3", "F4", "F5"]
    senders_benign = ["G1", "G2", "G3", "G4"]
    receivers_out = ["R1", "R2", "R3", "R4", "R5"]
    receivers_benign = ["Q1", "Q2", "Q3", "Q4"]
    rows = [
        f"2022/09/01 {9 + i:02d}:00,010,{s},010,T,20000.00,US Dollar,20000.00,US Dollar,Wire,0"
        for i, s in enumerate(senders_in)
    ] + [
        f"2022/09/01 {9 + i:02d}:00,010,{s},010,U,20000.00,US Dollar,20000.00,US Dollar,Wire,0"
        for i, s in enumerate(senders_benign)
    ] + [
        f"2022/09/01 {9 + i:02d}:00,010,S,010,{r},20000.00,US Dollar,20000.00,US Dollar,Wire,0"
        for i, r in enumerate(receivers_out)
    ] + [
        f"2022/09/01 {9 + i:02d}:00,010,S2,010,{r},20000.00,US Dollar,20000.00,US Dollar,Wire,0"
        for i, r in enumerate(receivers_benign)
    ] + [
        "2022/09/02 00:20,010,L1,010,L2,100000.00,US Dollar,100000.00,US Dollar,Wire,1"
    ]
    tx_rows = _rows(*rows)
    accounts = [
        *senders_in, "T", *senders_benign, "U", "S", *receivers_out,
        "S2", *receivers_benign, "L1", "L2",
    ]
    data_dir = _ingest_and_run_rules(monkeypatch, tmp_path, tx_rows, accounts)

    alerts = _alerts(data_dir)
    assert [(s, e) for s, e, _ in alerts] == [
        ("fan-in", "T"),
        ("fan-out", "S"),
    ]


# --- circular flow + dense community----------------------------------


def test_circular_flow_fires_on_cycles_and_open_chains_stay_silent(
    tmp_path, monkeypatch
) -> None:
    # Hand-computed (tolerance: back-amount within +/- 10% of the out-amount):
    # - A -> B 5000, B -> A 4900: 2-cycle preserved (4900 in [4500, 5500])
    #   -> 1 canonical alert, entity = smallest participant = A.
    # - E -> F 3000, F -> G 3100, G -> E 2900: 3-cycle preserved
    #   (2900 in [2700, 3300]) -> 1 canonical alert, entity = E.
    # - C -> D 5000, D -> C 2000: NOT amount-preserved (2000 < 4500) -> silent.
    # Cross-fire (correct behavior): round trips are pass-throughs, so the
    # churn scenario also fires for B (paid 98% of the 5000 received),
    # F (103%) and G (94%) — expected churn set {B, F, G}.
    tx_rows = _rows(
        "2022/09/01 00:20,010,L1,010,L2,100000.00,US Dollar,100000.00,US Dollar,Wire,1",
        "2022/09/01 09:00,010,A,010,B,5000.00,US Dollar,5000.00,US Dollar,Wire,0",
        "2022/09/01 10:00,010,B,010,A,4900.00,US Dollar,4900.00,US Dollar,Wire,0",
        "2022/09/01 11:00,010,C,010,D,5000.00,US Dollar,5000.00,US Dollar,Wire,0",
        "2022/09/01 12:00,010,D,010,C,2000.00,US Dollar,2000.00,US Dollar,Wire,0",
        "2022/09/01 13:00,010,E,010,F,3000.00,US Dollar,3000.00,US Dollar,Wire,0",
        "2022/09/01 14:00,010,F,010,G,3100.00,US Dollar,3100.00,US Dollar,Wire,0",
        "2022/09/01 15:00,010,G,010,E,2900.00,US Dollar,2900.00,US Dollar,Wire,0",
    )
    accounts = ["L1", "L2", "A", "B", "C", "D", "E", "F", "G"]
    data_dir = _ingest_and_run_rules(monkeypatch, tmp_path, tx_rows, accounts)

    alerts = _alerts(data_dir)
    assert [(s, e) for s, e, _ in alerts] == [
        ("circular-flow", "A"),
        ("circular-flow", "E"),
        ("rapid-churn", "B"),
        ("rapid-churn", "F"),
        ("rapid-churn", "G"),
    ]
    assert "path=A->B->A" in alerts[0][2]
    assert "path=E->F->G->E" in alerts[1][2]


def test_dense_community_fires_on_shared_counterparties_and_sparse_stays_silent(
    tmp_path, monkeypatch
) -> None:
    # Hand-computed (threshold: >= 5 shared counterparties in one day):
    # - M1 and M2 both transact with X1..X5 (100 each) on 09/01
    #   -> shared = 5 (boundary) -> dense-community alert (entity M1,
    #   pair=M1|M2). Amounts 500 < fan-out aggregate; 5 txs < velocity, so
    #   no cross-fires.
    # - M3 and M4 share only Y1, Y2 -> shared = 2, silent.
    shared = [f"X{i}" for i in range(1, 6)]
    sparse = ["Y1", "Y2"]
    rows = [
        f"2022/09/01 {9 + i:02d}:00,010,M1,010,{s},100.00,US Dollar,100.00,US Dollar,ACH,0"
        for i, s in enumerate(shared)
    ] + [
        f"2022/09/01 {9 + i:02d}:30,010,M2,010,{s},100.00,US Dollar,100.00,US Dollar,ACH,0"
        for i, s in enumerate(shared)
    ] + [
        "2022/09/01 15:00,010,M3,010,Y1,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/01 15:30,010,M4,010,Y1,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/01 16:00,010,M3,010,Y2,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/01 16:30,010,M4,010,Y2,100.00,US Dollar,100.00,US Dollar,ACH,0",
        "2022/09/02 00:20,010,L1,010,L2,100000.00,US Dollar,100000.00,US Dollar,Wire,1",
    ]
    tx_rows = _rows(*rows)
    accounts = ["M1", "M2", "M3", "M4", *shared, *sparse, "L1", "L2"]
    data_dir = _ingest_and_run_rules(monkeypatch, tmp_path, tx_rows, accounts)

    alerts = _alerts(data_dir)
    assert [(s, e) for s, e, _ in alerts] == [("dense-community", "M1")]
    assert "pair=M1|M2" in alerts[0][2]
    assert "shared_counterparties=5" in alerts[0][2]
