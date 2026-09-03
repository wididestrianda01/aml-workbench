"""C2 + C3 + C4 — typed deterministic DuckDB ingest with fail-closed gates.

Order of operations per track (fail-closed before any output is written):
1. C2: verify every raw file against the frozen manifest (SHA-256 + bytes).
2. C4: assert expected counts / integrity via in-memory queries over the CSVs.
3. C3: only then write typed DuckDB tables and export canonical parquet with a
   total, data-defined row order — re-running on identical inputs produces
   byte-identical parquet.

Schema conventions: Elliptic txId VARCHAR (hash-like, never integer),
time_step SMALLINT 1-49, class map 1 -> illicit / 2 -> licit / unknown -> NULL.
HI-Small: bank/account ids VARCHAR, typed TIMESTAMP, laundering flag 0/1.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from aml_workbench import config
from aml_workbench.data.gates import (
    assert_class_feature_id_sets_equal,
    assert_edge_referential_integrity,
    assert_elliptic_counts,
    assert_hi_small_counts,
)
from aml_workbench.data.manifest import load_manifest, pins_for, verify_raw_files
from aml_workbench.errors import DataQualityError

ELLIPTIC_FEATURE_COLUMNS: dict[str, str] = {
    "txId": "VARCHAR",
    "time_step": "SMALLINT",
    **{f"f{i:03d}": "REAL" for i in range(1, config.FEATURE_COUNT + 1)},
}
ELLIPTIC_CLASS_COLUMNS: dict[str, str] = {"txId": "VARCHAR", "class": "VARCHAR"}
ELLIPTIC_EDGE_COLUMNS: dict[str, str] = {"txId1": "VARCHAR", "txId2": "VARCHAR"}

HI_SMALL_TRANS_COLUMNS: dict[str, str] = {
    "ts": "VARCHAR",
    "from_bank": "VARCHAR",
    "from_account": "VARCHAR",
    "to_bank": "VARCHAR",
    "to_account": "VARCHAR",
    "amount_received": "DOUBLE",
    "receiving_currency": "VARCHAR",
    "amount_paid": "DOUBLE",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "UTINYINT",
}
HI_SMALL_ACCOUNT_COLUMNS: dict[str, str] = {
    "bank_name": "VARCHAR",
    "bank_id": "VARCHAR",
    "account_number": "VARCHAR",
    "entity_id": "VARCHAR",
    "entity_name": "VARCHAR",
}

HI_SMALL_TIME_FORMAT = "%Y/%m/%d %H:%M"


def _columns_sql(cols: dict[str, str]) -> str:
    return ", ".join(f"'{name}': '{dtype}'" for name, dtype in cols.items())


def _read_csv(path: Path, cols: dict[str, str], *, header: bool) -> str:
    return (
        f"read_csv('{path.as_posix()}', header={'true' if header else 'false'}, "
        f"columns={{{_columns_sql(cols)}}})"
    )


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None or row[0] is None:
        raise DataQualityError(f"Gate query returned no value: {sql}")
    return int(row[0])


# --- Elliptic -----------------------------------------------------------------


def ingest_elliptic(data_dir: Path, db_path: Path) -> list[str]:
    raw_dir = data_dir / "raw" / "elliptic"
    # C2 — before anything is written.
    pins = pins_for(load_manifest(data_dir), "elliptic")
    verify_raw_files(pins, raw_dir, "elliptic")

    features_csv = _read_csv(
        raw_dir / "elliptic_txs_features.csv", ELLIPTIC_FEATURE_COLUMNS, header=False
    )
    classes_csv = _read_csv(
        raw_dir / "elliptic_txs_classes.csv", ELLIPTIC_CLASS_COLUMNS, header=True
    )
    edges_csv = _read_csv(
        raw_dir / "elliptic_txs_edgelist.csv", ELLIPTIC_EDGE_COLUMNS, header=True
    )

    # C4 — count assertions against in-memory reads, before any artifact is written.
    mem = duckdb.connect(":memory:")
    try:
        tx_count = _scalar(mem, f"SELECT count(*) FROM {features_csv}")
        class_rows = mem.execute(
            f"SELECT class, count(*) FROM {classes_csv} GROUP BY class"
        ).fetchall()
        class_counts: dict[int | None, int] = {}
        for raw_class, count in class_rows:
            key: int | None
            if raw_class == "1":
                key = 1
            elif raw_class == "2":
                key = 2
            else:
                key = None if raw_class == "unknown" else (int(raw_class) if raw_class else None)
            class_counts[key] = int(count)
        step_rows = mem.execute(
            f"SELECT DISTINCT time_step FROM {features_csv}"
        ).fetchall()
        steps = {int(r[0]) for r in step_rows}
        edge_count = _scalar(mem, f"SELECT count(*) FROM {edges_csv}")
        orphan_endpoints = _scalar(
            mem,
            f"""
            WITH e AS (SELECT txId1 AS a, txId2 AS b FROM {edges_csv}),
                 f AS (SELECT txId FROM {features_csv})
            SELECT count(*) FROM (
                SELECT a AS tx FROM e LEFT JOIN f ON e.a = f.txId WHERE f.txId IS NULL
                UNION ALL
                SELECT b AS tx FROM e LEFT JOIN f ON e.b = f.txId WHERE f.txId IS NULL
            )
            """,
        )
        only_in_classes = _scalar(
            mem,
            f"SELECT count(*) FROM "
            f"(SELECT txId FROM {classes_csv} EXCEPT SELECT txId FROM {features_csv})",
        )
        only_in_features = _scalar(
            mem,
            f"SELECT count(*) FROM "
            f"(SELECT txId FROM {features_csv} EXCEPT SELECT txId FROM {classes_csv})",
        )
    finally:
        mem.close()

    assert_elliptic_counts(tx_count, edge_count, class_counts, steps)
    assert_edge_referential_integrity(orphan_endpoints)
    assert_class_feature_id_sets_equal(only_in_classes, only_in_features)

    # C3 — typed tables with total, data-defined ordering; CREATE OR REPLACE keeps
    # re-runs idempotent (never duplicate rows).
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE elliptic_tx AS
            SELECT txId AS tx_id,
                   CAST(CASE WHEN class = '1' THEN 1
                             WHEN class = '2' THEN 2
                             ELSE NULL END AS TINYINT) AS class_label
            FROM {classes_csv}
            ORDER BY txId
            """
        )
        feature_cols = ", ".join(f"f{i:03d}" for i in range(1, config.FEATURE_COUNT + 1))
        con.execute(
            f"""
            CREATE OR REPLACE TABLE elliptic_tx_features AS
            SELECT txId AS tx_id, time_step, {feature_cols}
            FROM {features_csv}
            ORDER BY txId
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE elliptic_edge AS
            SELECT txId1 AS src_tx_id, txId2 AS dst_tx_id
            FROM {edges_csv}
            ORDER BY src_tx_id, dst_tx_id
            """
        )
        _export_parquet(
            con,
            data_dir / "ingest" / "elliptic",
            {
                "tx.parquet": "SELECT * FROM elliptic_tx ORDER BY tx_id",
                "tx_features.parquet": (
                    "SELECT * FROM elliptic_tx_features ORDER BY tx_id"
                ),
                "edges.parquet": "SELECT * FROM elliptic_edge ORDER BY src_tx_id, dst_tx_id",
            },
        )
    finally:
        con.close()

    return [
        f"elliptic: {tx_count} tx / {edge_count} edges / "
        f"classes {class_counts.get(1)} illicit, {class_counts.get(2)} licit, "
        f"{class_counts.get(None)} unknown / steps {min(steps)}..{max(steps)}",
    ]


# --- HI-Small -----------------------------------------------------------------


def ingest_hismall(data_dir: Path, db_path: Path) -> list[str]:
    raw_dir = data_dir / "raw" / "hi-small"
    pins = pins_for(load_manifest(data_dir), "hi-small")
    verify_raw_files(pins, raw_dir, "hi-small")
    trans_csv = _read_csv(
        raw_dir / "HI-Small_Trans.csv", HI_SMALL_TRANS_COLUMNS, header=True
    )
    accounts_csv = _read_csv(
        raw_dir / "HI-Small_accounts.csv", HI_SMALL_ACCOUNT_COLUMNS, header=True
    )

    mem = duckdb.connect(":memory:")
    try:
        tx_count = _scalar(mem, f"SELECT count(*) FROM {trans_csv}")
        account_count = _scalar(
            mem,
            f"""
            SELECT count(*) FROM (
                SELECT from_bank, from_account FROM {trans_csv}
                UNION
                SELECT to_bank, to_account FROM {trans_csv}
            )
            """,
        )
        laundering_count = _scalar(
            mem, f"SELECT COALESCE(sum(is_laundering), 0) FROM {trans_csv}"
        )
    finally:
        mem.close()

    assert_hi_small_counts(tx_count, account_count, laundering_count)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE hismall_transaction AS
            SELECT strptime(ts, '{HI_SMALL_TIME_FORMAT}') AS tx_time,
                   from_bank, from_account, to_bank, to_account,
                   amount_received, receiving_currency, amount_paid,
                   payment_currency, payment_format, is_laundering
            FROM {trans_csv}
            ORDER BY ALL
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE hismall_account AS
            SELECT bank_name, bank_id, account_number, entity_id, entity_name
            FROM {accounts_csv}
            ORDER BY ALL
            """
        )
        _export_parquet(
            con,
            data_dir / "ingest" / "hi-small",
            {
                "transactions.parquet": "SELECT * FROM hismall_transaction ORDER BY ALL",
                "accounts.parquet": "SELECT * FROM hismall_account ORDER BY ALL",
            },
        )
    finally:
        con.close()

    laundering_rate = laundering_count / tx_count if tx_count else 0.0
    return [
        f"hi-small: {tx_count} tx / {account_count} accounts / "
        f"{laundering_count} flagged (rate {laundering_rate:.6f}, 1-in-{1 / laundering_rate:.1f})"
    ]


# --- shared --------------------------------------------------------------------


def _export_parquet(
    con: duckdb.DuckDBPyConnection, export_dir: Path, queries: dict[str, str]
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for file_name, query in queries.items():
        dest = (export_dir / file_name).as_posix()
        con.execute(f"COPY ({query}) TO '{dest}' (FORMAT PARQUET)")


def run_ingest(data_dir: Path, track: str = "all") -> list[str]:
    """Ingest entry point used by the CLI; dispatches on track."""
    db_path = data_dir / "workbench.duckdb"
    stats: list[str] = []
    if track in ("elliptic", "all"):
        stats += ingest_elliptic(data_dir, db_path)
    if track in ("hi-small", "all"):
        stats += ingest_hismall(data_dir, db_path)
    return stats
