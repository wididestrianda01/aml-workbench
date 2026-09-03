"""P1-05 seam tests: C4 Elliptic count assertions, fail-closed.

Tests inject violations by monkeypatching the frozen config constants at
runtime — never by editing them.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app
from conftest import build_elliptic_fixture

runner = CliRunner()


def _ingest(data_dir):
    return runner.invoke(app, ["ingest", "--data-dir", str(data_dir), "--track", "elliptic"])


def _patch_expected(monkeypatch, fixture) -> None:
    monkeypatch.setattr(config, "EXPECTED_TX_COUNT", len(fixture.tx_ids))
    monkeypatch.setattr(config, "EXPECTED_EDGE_COUNT", len(fixture.edges))
    monkeypatch.setattr(config, "EXPECTED_CLASS_COUNTS", fixture.class_counts)
    monkeypatch.setattr(config, "EXPECTED_TIME_STEPS", frozenset(fixture.steps))


def test_wrong_tx_count_exits_nonzero_no_outputs(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    monkeypatch.setattr(config, "EXPECTED_TX_COUNT", len(fixture.tx_ids) + 1)
    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1
    assert not (elliptic_data_dir / "workbench.duckdb").exists()
    assert not (elliptic_data_dir / "ingest").exists()


def test_wrong_edge_count_exits_nonzero(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    monkeypatch.setattr(config, "EXPECTED_EDGE_COUNT", len(fixture.edges) - 1)
    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1


def test_wrong_class_distribution_exits_nonzero(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    counts = fixture.class_counts
    counts[1] = counts[1] - 1  # one fewer illicit
    monkeypatch.setattr(config, "EXPECTED_CLASS_COUNTS", counts)
    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1


def test_missing_time_step_exits_nonzero(elliptic_data_dir, monkeypatch) -> None:
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    monkeypatch.setattr(config, "EXPECTED_TIME_STEPS", frozenset(fixture.steps) | {99})
    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1


def test_unknown_edge_endpoint_exits_nonzero(elliptic_data_dir, monkeypatch) -> None:
    """An edge endpoint not present in the txId set must stop the pipeline."""
    fixture = build_elliptic_fixture()
    _patch_expected(monkeypatch, fixture)
    edges_path = elliptic_data_dir / "raw" / "elliptic" / "elliptic_txs_edgelist.csv"
    lines = edges_path.read_text().splitlines()
    lines.append("999999999,555555555")  # both endpoints unknown
    edges_path.write_text("\n".join(lines) + "\n")
    # The manifest is frozen over the original file: recompute the pin so ONLY
    # the referential-integrity gate (not the checksum gate) fires.
    import hashlib
    import json

    data = edges_path.read_bytes()
    manifest_path = elliptic_data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["datasets"]["elliptic"]["files"]["elliptic_txs_edgelist.csv"]
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    result = _ingest(elliptic_data_dir)
    assert result.exit_code == 1
    assert not (elliptic_data_dir / "ingest").exists()


def test_expected_constants_are_the_locked_numbers() -> None:
    """The frozen constants are the numbers the portfolio cites."""
    assert config.EXPECTED_TX_COUNT == 203_769
    assert config.EXPECTED_EDGE_COUNT == 234_355
    assert config.EXPECTED_CLASS_COUNTS == {1: 4_545, 2: 42_019, None: 157_205}
    assert config.EXPECTED_TIME_STEPS == frozenset(range(1, 50))


@pytest.mark.parametrize(
    ("actual", "expected", "label"),
    [(203_768, 203_769, "elliptic transactions")],
)
def test_assert_count_message(actual: int, expected: int, label: str) -> None:
    from aml_workbench.data.gates import assert_count
    from aml_workbench.errors import DataQualityError

    try:
        assert_count(actual, expected, label)
    except DataQualityError as exc:
        assert str(expected) in str(exc) and str(actual) in str(exc)
    else:
        raise AssertionError("expected DataQualityError")
