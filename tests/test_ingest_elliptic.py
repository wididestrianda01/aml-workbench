"""P1-03 + P1-04 seam tests: Elliptic typed ingest, C2 checksum gate, C3
determinism (byte-identical parquet re-run), idempotency.

All tests drive the CLI (pipeline-command seam) against a tmp_path DuckDB and
assert external behavior only.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app
from conftest import build_elliptic_fixture

runner = CliRunner()


def _ingest(data_dir: Path):
    return runner.invoke(app, ["ingest", "--data-dir", str(data_dir), "--track", "elliptic"])


def _patch_expected(monkeypatch, fixture) -> None:
    monkeypatch.setattr(config, "EXPECTED_TX_COUNT", len(fixture.tx_ids))
    monkeypatch.setattr(config, "EXPECTED_EDGE_COUNT", len(fixture.edges))
    monkeypatch.setattr(config, "EXPECTED_CLASS_COUNTS", fixture.class_counts)
    monkeypatch.setattr(config, "EXPECTED_TIME_STEPS", frozenset(fixture.steps))


def _parquet_checksums(data_dir: Path) -> dict[str, str]:
    import hashlib

    export = data_dir / "ingest" / "elliptic"
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(export.glob("*.parquet"))
    }


def test_green_path_types_schema_and_artifacts(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 0, result.output

    import duckdb

    con = duckdb.connect(str(elliptic_data_dir / "workbench.duckdb"), read_only=True)
    try:
        # txId VARCHAR (hash-like, never integer), time_step SMALLINT, class NULL for unknown.
        cols = {
            r[1]: r[2] for r in con.execute("PRAGMA table_info('elliptic_tx')").fetchall()
        }
        assert cols["tx_id"].upper() == "VARCHAR"
        assert cols["class_label"].upper() == "TINYINT"
        feature_cols = {
            r[1]: r[2]
            for r in con.execute("PRAGMA table_info('elliptic_tx_features')").fetchall()
        }
        assert feature_cols["f001"].upper() in {"REAL", "FLOAT"}
        assert feature_cols["time_step"].upper() == "SMALLINT"
        rows = con.execute("SELECT tx_id, class_label FROM elliptic_tx ORDER BY tx_id").fetchall()
    finally:
        con.close()
    by_id = dict(rows)
    assert len(rows) == len(fixture.tx_ids)
    unknown_ids = {
        t for t, label in zip(fixture.tx_ids, fixture.labels, strict=True)
        if label == "unknown"
    }
    for tx_id in unknown_ids:
        assert by_id[tx_id] is None, "unknown class must be NULL"
    illicit_ids = {
        t for t, label in zip(fixture.tx_ids, fixture.labels, strict=True)
        if label == "1"
    }
    for tx_id in illicit_ids:
        assert by_id[tx_id] == 1
    # Parquet artifacts exist.
    assert {p.name for p in (elliptic_data_dir / "ingest" / "elliptic").glob("*.parquet")} == {
        "tx.parquet",
        "tx_features.parquet",
        "edges.parquet",
    }


def test_tampered_checksum_fails_closed_before_any_output(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    # Tamper the manifest checksum (never the file on disk — ingest must refuse
    # against the frozen pin).
    manifest_path = elliptic_data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["datasets"]["elliptic"]["files"]["elliptic_txs_features.csv"]
    entry["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1
    assert not (elliptic_data_dir / "workbench.duckdb").exists()
    assert not (elliptic_data_dir / "ingest").exists()


def test_byte_size_mismatch_alone_refuses(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    manifest_path = elliptic_data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["datasets"]["elliptic"]["files"]["elliptic_txs_classes.csv"]
    correct = entry["bytes"]
    entry["bytes"] = correct + 1
    manifest_path.write_text(json.dumps(manifest))

    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1
    assert not (elliptic_data_dir / "workbench.duckdb").exists()


def test_determinism_rerun_byte_identical_parquet(elliptic_data_dir, monkeypatch) -> None:
    """C3 seam test: ingest twice on a frozen fixture -> parquet checksums equal."""
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    assert _ingest(elliptic_data_dir).exit_code == 0
    first = _parquet_checksums(elliptic_data_dir)
    assert len(first) == 3
    assert _ingest(elliptic_data_dir).exit_code == 0
    second = _parquet_checksums(elliptic_data_dir)
    assert first == second


def test_ingest_idempotent_no_duplicate_rows(elliptic_data_dir, monkeypatch) -> None:
    import duckdb

    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    assert _ingest(elliptic_data_dir).exit_code == 0
    assert _ingest(elliptic_data_dir).exit_code == 0
    con = duckdb.connect(str(elliptic_data_dir / "workbench.duckdb"), read_only=True)
    try:
        n = con.execute("SELECT count(*) FROM elliptic_tx").fetchone()[0]
        edges = con.execute("SELECT count(*) FROM elliptic_edge").fetchone()[0]
        distinct = con.execute("SELECT count(DISTINCT tx_id) FROM elliptic_tx").fetchone()[0]
    finally:
        con.close()
    assert n == len(fixture.tx_ids)  # re-run replaced, never duplicated
    assert distinct == n
    assert edges == len(fixture.edges)
