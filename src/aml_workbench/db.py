"""Shared DuckDB store access: one open-with-table-gates and one scalar read.

Every pipeline stage opens the same workbench.duckdb; the fail-closed checks
(l database exists, required tables present) live here once so stages cannot
drift apart.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from aml_workbench.errors import DataQualityError


def open_workbench(
    data_dir: Path,
    required_tables: dict[str, str],
    *,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open workbench.duckdb, asserting every required table exists.

    Fail-closed: a missing database or any missing table raises
    DataQualityError before the caller runs a single query. required_tables
    maps table name -> remediation hint shown when the table is absent.
    """
    db_path = data_dir / "workbench.duckdb"
    if not db_path.exists():
        raise DataQualityError(
            f"workbench database not found at {db_path}; run `aml ingest` first"
        )
    con = duckdb.connect(str(db_path), read_only=read_only)
    tables = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    missing = {name: hint for name, hint in required_tables.items() if name not in tables}
    if missing:
        con.close()
        detail = "; ".join(f"{name} ({hint})" for name, hint in sorted(missing.items()))
        raise DataQualityError(
            f"workbench database {db_path} lacks required table(s): {detail}"
        )
    return con


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Fetch a single integer scalar; fail-closed on absence."""
    row = con.execute(sql).fetchone()
    if row is None or row[0] is None:
        raise DataQualityError(f"Gate query returned no value: {sql}")
    return int(row[0])
