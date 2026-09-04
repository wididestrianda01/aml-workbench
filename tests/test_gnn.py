"""Seam tests: strict-inductive GraphSAGE GNN baseline and the honest
GNN-vs-GBM comparison. Drives `aml gnn` against a tmp_path DuckDB seeded with
the synthetic Elliptic fixture plus a small edge table; external behavior only.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from aml_workbench import config
from conftest import run_stages, seed_workbench

_PROTOCOL_SNIPPET = "no test-period adjacency"


def _seed_edges(db_path: Path) -> None:
    """Chain edges within the train window plus one test-window edge; the
    inductive edge mask must drop the latter (asserted by the run, not here)."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE elliptic_edge (src_tx_id VARCHAR, dst_tx_id VARCHAR)")
        rows = [(f"{i:09d}", f"{i + 1:09d}") for i in range(33)]  # steps 1-34 chain
        rows.append((f"{35:09d}", f"{36:09d}"))  # test-window edge: must be dropped
        con.executemany("INSERT INTO elliptic_edge VALUES (?, ?)", rows)
    finally:
        con.close()


def _seed_challenger_metrics(tmp_path: Path) -> None:
    """Minimal challenger artifact so the comparison stage can join."""
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "challenger_metrics.json").write_text(
        json.dumps(
            {
                "seeds": list(config.MODEL_SEEDS),
                "metrics": [
                    {"model": "lightgbm_challenger", "seed": s, "roc_auc": 0.9, "pr_auc": 0.5}
                    for s in config.MODEL_SEEDS
                ],
            }
        ),
        encoding="utf-8",
    )


def test_gnn_trains_3_seeds_and_reports_comparison(tmp_path: Path) -> None:
    seed_workbench(tmp_path / "workbench.duckdb")
    _seed_edges(tmp_path / "workbench.duckdb")
    _seed_challenger_metrics(tmp_path)
    result = run_stages("gnn", data_dir=tmp_path)
    assert result.exit_code == 0, result.output

    payload = json.loads(
        (tmp_path / "reports" / "gnn_metrics.json").read_text(encoding="utf-8")
    )
    assert {m["seed"] for m in payload["per_seed"]} == set(config.MODEL_SEEDS)
    for m in payload["per_seed"]:
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["pr_auc"] <= 1.0
    assert _PROTOCOL_SNIPPET in payload["protocol"]

    comparison = json.loads(
        (tmp_path / "reports" / "gnn_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["gnn"]["seeds"] == list(config.MODEL_SEEDS)
    assert comparison["challenger"]["seeds"] == list(config.MODEL_SEEDS)
    assert "verdict" in comparison
    assert _PROTOCOL_SNIPPET in comparison["protocol"]
    report = (tmp_path / "reports" / "gnn_comparison.md").read_text(encoding="utf-8")
    assert "GraphSAGE" in report and "LightGBM challenger" in report


def test_gnn_without_challenger_fail_closed(tmp_path: Path) -> None:
    seed_workbench(tmp_path / "workbench.duckdb")
    _seed_edges(tmp_path / "workbench.duckdb")
    (tmp_path / "reports").mkdir()
    result = run_stages("gnn", data_dir=tmp_path)
    assert result.exit_code == 1, result.output
    assert "Fail-closed" in result.output
    assert not (tmp_path / "reports" / "gnn_metrics.json").exists()
