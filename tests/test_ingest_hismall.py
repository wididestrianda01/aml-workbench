"""P1-06 seam tests: HI-Small ingest — typed schema, manifest gate, count
gates, determinism. Same pattern as the Elliptic tests: CLI seam, tmp_path,
violation injection via monkeypatched config constants.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from aml_workbench import config
from conftest import build_hismall_fixture, parquet_checksums, run_ingest

runner = CliRunner()


def _ingest(data_dir: Path):
    return run_ingest(data_dir, "hi-small")


def _patch_expected(
    monkeypatch, *, tx: int, accounts: int, laundering: int, rate_target: float
) -> None:
    monkeypatch.setattr(config, "HI_SMALL_MIN_TX", tx)
    monkeypatch.setattr(config, "HI_SMALL_MIN_ACCOUNTS", accounts)
    monkeypatch.setattr(config, "HI_SMALL_LAUNDERING_COUNT_PINNED", laundering)
    monkeypatch.setattr(config, "HI_SMALL_LAUNDERING_RATE_TARGET", rate_target)


def test_types_and_counts(hismall_data_dir, monkeypatch) -> None:
    fixture = build_hismall_fixture()
    _patch_expected(
        monkeypatch,
        tx=len(fixture.tx_rows),
        accounts=3,  # distinct (bank, account) pairs across from/to
        laundering=fixture.laundering_count,
        rate_target=fixture.laundering_count / len(fixture.tx_rows),
    )
    result = _ingest(hismall_data_dir)
    assert result.exit_code == 0, result.output

    import duckdb

    con = duckdb.connect(str(hismall_data_dir / "workbench.duckdb"), read_only=True)
    try:
        cols = {
            r[1]: r[2]
            for r in con.execute("PRAGMA table_info('hismall_transaction')").fetchall()
        }
        assert cols["from_bank"].upper() == "VARCHAR"
        assert cols["from_account"].upper() == "VARCHAR"
        assert cols["to_account"].upper() == "VARCHAR"
        assert cols["tx_time"].upper() == "TIMESTAMP"
        assert cols["is_laundering"].upper() == "UTINYINT"
        n = con.execute("SELECT count(*) FROM hismall_transaction").fetchone()[0]
        flagged = con.execute(
            "SELECT count(*) FROM hismall_transaction WHERE is_laundering = 1"
        ).fetchone()[0]
        ts_type_check = con.execute(
            "SELECT typeof(tx_time) FROM hismall_transaction LIMIT 1"
        ).fetchone()[0]
        accounts_n = con.execute("SELECT count(*) FROM hismall_account").fetchone()[0]
    finally:
        con.close()
    assert n == len(fixture.tx_rows)
    assert flagged == fixture.laundering_count
    assert accounts_n == len(fixture.account_rows)
    assert "TIMESTAMP" in ts_type_check


def test_bad_checksum_fails_closed(hismall_data_dir, monkeypatch) -> None:
    fixture = build_hismall_fixture()
    _patch_expected(
        monkeypatch,
        tx=len(fixture.tx_rows),
        accounts=3,
        laundering=fixture.laundering_count,
        rate_target=fixture.laundering_count / len(fixture.tx_rows),
    )
    manifest_path = hismall_data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["datasets"]["hi-small"]["files"]["HI-Small_Trans.csv"]["sha256"] = "1" * 64
    manifest_path.write_text(json.dumps(manifest))

    result = _ingest(hismall_data_dir)
    assert result.exit_code == 1
    assert not (hismall_data_dir / "workbench.duckdb").exists()
    assert not (hismall_data_dir / "ingest").exists()


def test_wrong_tx_count_fails_closed(hismall_data_dir, monkeypatch) -> None:
    fixture = build_hismall_fixture()
    _patch_expected(
        monkeypatch,
        tx=len(fixture.tx_rows) + 1,
        accounts=3,
        laundering=fixture.laundering_count,
        rate_target=fixture.laundering_count / len(fixture.tx_rows),
    )
    result = _ingest(hismall_data_dir)
    assert result.exit_code == 1
    assert not (hismall_data_dir / "ingest").exists()


def test_laundering_rate_drift_fails_closed(hismall_data_dir, monkeypatch) -> None:
    fixture = build_hismall_fixture()
    _patch_expected(
        monkeypatch,
        tx=len(fixture.tx_rows),
        accounts=3,
        laundering=fixture.laundering_count,
        rate_target=1 / 981,  # real-world target; fixture rate is far off
    )
    result = _ingest(hismall_data_dir)
    assert result.exit_code == 1


def test_laundering_count_drift_from_pin_fails_closed(hismall_data_dir, monkeypatch) -> None:
    fixture = build_hismall_fixture()
    _patch_expected(
        monkeypatch,
        tx=len(fixture.tx_rows),
        accounts=3,
        laundering=fixture.laundering_count + 1,  # drifted pin
        rate_target=fixture.laundering_count / len(fixture.tx_rows),
    )
    result = _ingest(hismall_data_dir)
    assert result.exit_code == 1


def test_determinism_rerun_byte_identical_parquet(hismall_data_dir, monkeypatch) -> None:
    fixture = build_hismall_fixture()
    _patch_expected(
        monkeypatch,
        tx=len(fixture.tx_rows),
        accounts=3,
        laundering=fixture.laundering_count,
        rate_target=fixture.laundering_count / len(fixture.tx_rows),
    )
    assert _ingest(hismall_data_dir).exit_code == 0
    first = parquet_checksums(hismall_data_dir, 'hi-small')
    assert set(first) == {"transactions.parquet", "accounts.parquet"}
    assert _ingest(hismall_data_dir).exit_code == 0
    assert first == parquet_checksums(hismall_data_dir, 'hi-small')
