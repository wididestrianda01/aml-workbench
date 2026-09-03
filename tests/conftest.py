"""Shared fixtures: schema-faithful frozen extracts of real source rows.

These fixtures mirror the exact raw-file schemas (Elliptic: headerless 167-col
features, headered classes/edgelist; HI-Small: 11-col trans, 5-col accounts) so
tests drive the real pipeline commands against tmp_path stores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from aml_workbench import config

FEATURE_COUNT = config.FEATURE_COUNT

TRANS_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
)
ACCOUNTS_HEADER = "Bank Name,Bank ID,Account Number,Entity ID,Entity Name\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    data_dir: Path,
    dataset: str,
    files: dict[str, bytes],
    *,
    license_note: str,
    source_note: str,
) -> None:
    manifest = {
        "version": 1,
        "datasets": {
            dataset: {
                "license": license_note,
                "source": source_note,
                "files": {
                    name: {"bytes": len(data), "sha256": _sha(data), "channel": "fixture"}
                    for name, data in files.items()
                },
            }
        },
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --- Elliptic fixture -----------------------------------------------------------


class EllipticFixture:
    def __init__(
        self,
        tx_ids: list[str],
        steps: list[int],
        labels: list[str],
        edges: list[tuple[str, str]],
    ):
        self.tx_ids = tx_ids
        self.steps = steps
        self.labels = labels
        self.edges = edges

    @property
    def class_counts(self) -> dict[int | None, int]:
        counts: dict[int | None, int] = {1: 0, 2: 0, None: 0}
        for label in self.labels:
            key = {"1": 1, "2": 2}.get(label, None)
            counts[key] += 1
        return counts


def build_elliptic_fixture(seed: int = 0) -> EllipticFixture:
    rng = np.random.default_rng(seed)
    tx_ids = [f"{rng.integers(10**8, 10**9)}" for _ in range(6)]
    steps = [1, 1, 2, 2, 3, 3]
    labels = ["1", "2", "1", "2", "unknown", "unknown"]
    edges = [
        (tx_ids[0], tx_ids[1]),
        (tx_ids[1], tx_ids[2]),
        (tx_ids[2], tx_ids[3]),
        (tx_ids[3], tx_ids[0]),
    ]
    del rng
    return EllipticFixture(tx_ids, steps, labels, edges)


def elliptic_files(fixture: EllipticFixture) -> dict[str, bytes]:
    rng = np.random.default_rng(7)
    feature_lines = []
    for tx_id, step in zip(fixture.tx_ids, fixture.steps, strict=True):
        values = ",".join(f"{v:.8f}" for v in rng.normal(size=FEATURE_COUNT))
        feature_lines.append(f"{tx_id},{step},{values}")
    features = ("\n".join(feature_lines) + "\n").encode()
    class_lines = (
        f"{tx_id},{label}"
        for tx_id, label in zip(fixture.tx_ids, fixture.labels, strict=True)
    )
    classes = ("txId,class\n" + "\n".join(class_lines) + "\n").encode()
    edge_lines = (f"{a},{b}" for a, b in fixture.edges)
    edges = ("txId1,txId2\n" + "\n".join(edge_lines) + "\n").encode()
    return {
        "elliptic_txs_features.csv": features,
        "elliptic_txs_classes.csv": classes,
        "elliptic_txs_edgelist.csv": edges,
    }


@pytest.fixture
def elliptic_data_dir(tmp_path: Path) -> Path:
    """tmp data_dir with raw elliptic files + a matching frozen manifest."""
    data_dir = tmp_path / "data"
    fixture = build_elliptic_fixture()
    files = elliptic_files(fixture)
    raw_dir = data_dir / "raw" / "elliptic"
    raw_dir.mkdir(parents=True)
    for name, data in files.items():
        (raw_dir / name).write_bytes(data)
    _write_manifest(data_dir, "elliptic", files, license_note="fixture", source_note="fixture")
    return data_dir


# --- HI-Small fixture -----------------------------------------------------------


class HiSmallFixture:
    def __init__(self, tx_rows: list[list[str]], account_rows: list[list[str]]):
        self.tx_rows = tx_rows
        self.account_rows = account_rows

    @property
    def laundering_count(self) -> int:
        return sum(1 for row in self.tx_rows if row[-1] == "1")


def build_hismall_fixture() -> HiSmallFixture:
    tx_rows = [
        ["2022/09/01 00:20", "010", "8000EBD30", "010", "8000EBD30",
         "3697.34", "US Dollar", "3697.34", "US Dollar", "ACH", "0"],
        ["2022/09/01 01:05", "010", "8000EBD30", "0256398", "8148A8711",
         "0.281983", "Bitcoin", "0.281983", "Bitcoin", "Bitcoin", "1"],
        ["2022/09/02 12:00", "0256398", "8148A8711", "010", "8000EBD30",
         "100.00", "US Dollar", "100.00", "US Dollar", "Wire", "0"],
        ["2022/09/03 08:30", "0154518", "8148A6091", "0256398", "8148A8711",
         "50.00", "Euro", "55.00", "US Dollar", "Cheque", "0"],
    ]
    account_rows = [
        ["Portugal Bank #1", "010", "8000EBD30", "80062E240", "Entity A"],
        ["Portugal Bank #2", "0256398", "8148A8711", "80062E241", "Entity B"],
        ["Portugal Bank #3", "0154518", "8148A6091", "80062E242", "Entity C"],
    ]
    return HiSmallFixture(tx_rows, account_rows)


def hismall_files(fixture: HiSmallFixture) -> dict[str, bytes]:
    trans_body = "\n".join(",".join(row) for row in fixture.tx_rows)
    trans = (TRANS_HEADER + trans_body + "\n").encode()
    account_lines = "\n".join(",".join(row) for row in fixture.account_rows)
    accounts = (ACCOUNTS_HEADER + account_lines + "\n").encode()
    return {"HI-Small_Trans.csv": trans, "HI-Small_accounts.csv": accounts}


@pytest.fixture
def hismall_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    fixture = build_hismall_fixture()
    files = hismall_files(fixture)
    raw_dir = data_dir / "raw" / "hi-small"
    raw_dir.mkdir(parents=True)
    for name, data in files.items():
        (raw_dir / name).write_bytes(data)
    _write_manifest(data_dir, "hi-small", files, license_note="fixture", source_note="fixture")
    return data_dir


def make_hismall_data_dir(
    tmp_path: Path, tx_rows: list[list[str]], account_rows: list[list[str]]
) -> Path:
    """Scenario fixture: arbitrary constructed HI-Small rows written through
    the same raw-CSV + manifest path as the standard fixture, so rules tests
    exercise the full `ingest -> rules` pipeline seam on hand-built typologies.
    Gate expectations are patched to the fixture's own counts by the caller."""
    data_dir = tmp_path / "data"
    fixture = HiSmallFixture(tx_rows, account_rows)
    files = hismall_files(fixture)
    raw_dir = data_dir / "raw" / "hi-small"
    raw_dir.mkdir(parents=True)
    for name, data in files.items():
        (raw_dir / name).write_bytes(data)
    _write_manifest(data_dir, "hi-small", files, license_note="fixture", source_note="fixture")
    return data_dir


# --- shared seam helpers ---------------------------------------------------------


def run_ingest(data_dir, track):
    """CLI seam helper: ingest one track."""
    from typer.testing import CliRunner

    from aml_workbench.cli import app

    return CliRunner().invoke(
        app, ["ingest", "--data-dir", str(data_dir), "--track", track]
    )


def parquet_checksums(data_dir: Path, track: str) -> dict[str, str]:
    """sha256 of every exported parquet for a track (determinism assertions)."""
    import hashlib

    export = data_dir / "ingest" / track
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(export.glob("*.parquet"))
    }
