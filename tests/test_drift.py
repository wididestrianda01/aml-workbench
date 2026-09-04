"""Seam tests for the drift stage: PSI arithmetic hand-checked, threshold
breach flagged, missing artifacts fail closed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aml_workbench.drift import psi
from conftest import run_stages, seed_workbench


def test_psi_hand_computed() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5_000)
    # identical distributions -> zero shift
    assert abs(psi(x, x)) < 1e-6
    # hand-computed: expected [0.4, 0.6] vs actual [0.6, 0.4] over two bins
    # PSI = (0.6-0.4)*ln(0.6/0.4) + (0.4-0.6)*ln(0.4/0.6) = 0.4*ln(1.5)
    a = np.array([0.0] * 4 + [1.0] * 6)  # 40/60
    b = np.array([0.0] * 6 + [1.0] * 4)  # 60/40
    # hand-computed zero bin: reference 0.1/0.9, actual all in bin 1 with the
    # 1e-6 floor: PSI = (0-0.1)*ln(1e-6/0.1) + (1-0.9)*ln(1/0.9)
    c = np.array([0.0] + [1.0] * 9)
    d = np.array([1.0] * 10)
    manual = (1e-6 - 0.1) * np.log(1e-6 / 0.1) + (1.0 - 0.9) * np.log(1.0 / 0.9)
    assert abs(psi(a, b) - 0.4 * np.log(1.5)) < 1e-9
    assert abs(psi(c, d) - manual) < 1e-9



def test_psi_constant_reference_still_detects_drift() -> None:
    ref = np.ones(100)  # constant training column
    assert abs(psi(ref, np.ones(50))) < 1e-6  # unchanged -> zero
    assert psi(ref, np.ones(50) * 5.0) > 0.25  # shifted up -> breach
    assert psi(ref, np.zeros(50)) > 0.25  # shifted down -> breach

def test_drift_flags_breach_and_exits_nonzero(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    seed_workbench(data_dir / "workbench.duckdb", graph_only=False)
    # shift every test-step f001 far off the train distribution -> PSI breach
    import duckdb

    con = duckdb.connect(str(data_dir / "workbench.duckdb"))
    con.execute("UPDATE elliptic_tx_features SET f001 = f001 + 100 WHERE time_step >= 35")
    con.close()
    from aml_workbench.model import run_challenger

    run_challenger(data_dir)
    result = run_stages("drift", data_dir=data_dir)
    assert result.exit_code != 0, result.output
    metrics = json.loads((data_dir / "reports" / "drift_metrics.json").read_text())
    assert metrics["breached"] is True
    assert metrics["columns"]["f001"]["breach"] is True
    assert metrics["columns"]["f001"]["psi"] > 0.25


def test_drift_no_breach_exits_zero(tmp_path: Path, monkeypatch) -> None:
    from aml_workbench import config

    # the tiny fixture (136 train / 60 test rows) makes per-column PSI pure
    # sampling noise across ~172 columns, so the max exceeds any fixed
    # threshold; a raised breach threshold isolates the "no drift -> exit 0"
    # path (the breach path is covered by the shifted-f001 test above)
    monkeypatch.setattr(config, "PSI_BREACH_THRESHOLD", 5.0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    seed_workbench(data_dir / "workbench.duckdb", graph_only=False)
    from aml_workbench.model import run_challenger

    run_challenger(data_dir)
    result = run_stages("drift", data_dir=data_dir)
    assert result.exit_code == 0, result.output
    metrics = json.loads((data_dir / "reports" / "drift_metrics.json").read_text())
    assert metrics["breached"] is False


def test_drift_fail_closed_on_missing_model(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    seed_workbench(data_dir / "workbench.duckdb", graph_only=False)
    result = run_stages("drift", data_dir=data_dir)
    assert result.exit_code != 0
    assert "Fail-closed" in result.output
