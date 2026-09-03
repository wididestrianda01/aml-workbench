"""P3 seam tests: graph features over the Elliptic temporal graph.

Every test drives the pipeline seam (`aml ingest` -> `aml graph-features`)
against constructed graph fixtures with hand-computed expectations written
inline BEFORE the assertions run. Strict inductive: a tx at step t only sees
edges whose later endpoint has step <= t.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
from typer.testing import CliRunner

from aml_workbench import config
from aml_workbench.cli import app
from conftest import run_ingest, write_manifest

runner = CliRunner()
FEATURE_COUNT = config.FEATURE_COUNT


def _make_elliptic_dir(
    tmp_path: Path,
    tx_ids: list[str],
    steps: list[int],
    labels: list[str],
    edges: list[tuple[str, str]],
) -> Path:
    """Arbitrary constructed Elliptic raw files through the manifest + ingest seam."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw" / "elliptic"
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    feature_lines = [
        f"{tx},{step}," + ",".join(f"{v:.8f}" for v in rng.normal(size=FEATURE_COUNT))
        for tx, step in zip(tx_ids, steps, strict=True)
    ]
    files = {
        "elliptic_txs_features.csv": (
            "\n".join(feature_lines) + "\n"
        ).encode(),
        "elliptic_txs_classes.csv": (
            "txId,class\n"
            + "\n".join(f"{tx},{label}" for tx, label in zip(tx_ids, labels, strict=True))
            + "\n"
        ).encode(),
        "elliptic_txs_edgelist.csv": (
            "txId1,txId2\n" + "\n".join(f"{a},{b}" for a, b in edges) + "\n"
        ).encode(),
    }
    for name, data in files.items():
        (raw_dir / name).write_bytes(data)
    write_manifest(
        data_dir, "elliptic", files, license_note="fixture", source_note="fixture"
    )
    return data_dir


def _ingest_and_run(
    monkeypatch,
    tmp_path: Path,
    tx_ids: list[str],
    steps: list[int],
    labels: list[str],
    edges: list[tuple[str, str]],
) -> Path:
    data_dir = _make_elliptic_dir(tmp_path, tx_ids, steps, labels, edges)
    counts: dict[int | None, int] = {1: 0, 2: 0, None: 0}
    for label in labels:
        counts[{"1": 1, "2": 2}.get(label, None)] += 1
    counts = {k: v for k, v in counts.items() if v > 0}
    monkeypatch.setattr(config, "EXPECTED_TX_COUNT", len(tx_ids))
    monkeypatch.setattr(config, "EXPECTED_EDGE_COUNT", len(edges))
    monkeypatch.setattr(config, "EXPECTED_CLASS_COUNTS", counts)
    monkeypatch.setattr(config, "EXPECTED_TIME_STEPS", frozenset(steps))
    ingest = run_ingest(data_dir, "elliptic")
    assert ingest.exit_code == 0, ingest.output
    result = runner.invoke(app, ["graph-features", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    return data_dir


def _features(data_dir: Path) -> dict[str, tuple]:
    con = duckdb.connect(str(data_dir / "workbench.duckdb"), read_only=True)
    try:
        rows = con.execute(
            "SELECT tx_id, in_degree, out_degree, reciprocity, ego_illicit_1hop, "
            "ego_illicit_2hop, louvain_community, time_since_activity "
            "FROM tx_graph_features ORDER BY tx_id"
        ).fetchall()
    finally:
        con.close()
    return {row[0]: row[1:] for row in rows}


def _tx_ids(n: int) -> list[str]:
    return [str(200_000_000 + i) for i in range(n)]


# --- P3-01: seam + degrees -----------------------------------------------------


def test_graph_features_green_path_one_row_per_tx_hand_checked_degrees(
    tmp_path, monkeypatch
) -> None:
    # Fixture graph: 4-cycle t0->t1->t2->t3->t0, steps [1,1,2,2,3,3],
    # labels [illicit, licit, illicit, licit, unknown, unknown].
    # Edge visibility = later endpoint's step: t0->t1 at 1, t1->t2, t2->t3 and
    # t3->t0 (t3 has step 2) at 2. Hand-computed point-in-time degrees:
    #   t0 (step 1): sees only t0->t1           -> in 0, out 1
    #   t1 (step 1): sees only t0->t1           -> in 1, out 0
    #   t2 (step 2): full cycle now visible     -> in 1, out 1
    #   t3 (step 2): full cycle now visible     -> in 1, out 1
    #   t4, t5 (step 3): isolated               -> in 0, out 0
    t = _tx_ids(6)
    data_dir = _ingest_and_run(
        monkeypatch,
        tmp_path,
        t,
        [1, 1, 2, 2, 3, 3],
        ["1", "2", "1", "2", "unknown", "unknown"],
        [(t[0], t[1]), (t[1], t[2]), (t[2], t[3]), (t[3], t[0])],
    )
    feats = _features(data_dir)
    assert len(feats) == 6  # one row per tx, no more
    assert feats[t[0]][:2] == (0, 1)
    assert feats[t[1]][:2] == (1, 0)
    assert feats[t[2]][:2] == (1, 1)
    assert feats[t[3]][:2] == (1, 1)
    assert feats[t[4]][:2] == (0, 0)
    assert feats[t[5]][:2] == (0, 0)


def test_missing_ingest_tables_fail_closed(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = runner.invoke(app, ["graph-features", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "Fail-closed" in result.output


# --- P3-02: reciprocity + time-since-activity ----------------------------------


def test_reciprocity_and_time_since_activity_hand_checked(tmp_path, monkeypatch) -> None:
    # A(step 1, illicit), B(step 2, licit), C(step 3, licit).
    # Edges: A->B, B->A (mutual pair), C->B (one-way).
    # Visibility: A->B and B->A at step 2, C->B at step 3.
    #   A (step 1): graph empty at its step -> reciprocity 0.0, tsa 0
    #   B (step 2): incident {A->B, B->A}, both reciprocated -> 2/2 = 1.0;
    #               earlier neighbour A at step 1 -> tsa = 2 - 1 = 1
    #   C (step 3): incident {C->B}, reverse B->C absent      -> 0/1 = 0.0;
    #               earlier neighbour B at step 2 -> tsa = 3 - 2 = 1
    ids = ["310000001", "310000002", "310000003"]
    a, b, c = ids
    data_dir = _ingest_and_run(
        monkeypatch, tmp_path, ids, [1, 2, 3], ["1", "2", "2"], [(a, b), (b, a), (c, b)]
    )
    feats = _features(data_dir)
    assert feats[a][2] == 0.0
    assert feats[a][6] == 0  # first activity: no earlier neighbour, explicit 0
    assert feats[b][2] == 1.0
    assert feats[b][6] == 1
    assert feats[c][2] == 0.0
    assert feats[c][6] == 1


# --- P3-03: ego illicit fraction, strict inductive ------------------------------


def test_ego_illicit_fraction_unknowns_in_denominator_no_test_leak(
    tmp_path, monkeypatch
) -> None:
    # Hub H(step 30, illicit); I(step 10, illicit), L(step 20, licit),
    # U(step 25, unknown) all -> H. T(step 35, licit) -> H is a TEST-period
    # edge (visibility 35): it must not change H's features, frozen at step 30.
    #   H ego 1-hop: {I, L, U} -> 1 illicit / 3 = 1/3 (unknown stays in denom)
    #   H ego 2-hop: neighbourhood of I/L/U adds nothing new -> same set -> 1/3
    #   T ego 1-hop at step 35: {H} -> 1 illicit / 1 = 1.0
    h, i, lic, u, t = (
        "320000001",
        "320000002",
        "320000003",
        "320000004",
        "320000005",
    )
    data_dir = _ingest_and_run(
        monkeypatch,
        tmp_path,
        [h, i, lic, u, t],
        [30, 10, 20, 25, 35],
        ["1", "1", "2", "unknown", "2"],
        [(i, h), (lic, h), (u, h), (t, h)],
    )
    feats = _features(data_dir)
    assert feats[h][3] == 1 / 3
    assert feats[h][4] == 1 / 3
    assert feats[t][3] == 1.0


def test_two_hop_egohood_hand_checked(tmp_path, monkeypatch) -> None:
    # Chain X(step 1, illicit) -> Y(step 2, illicit) -> Z(step 3, licit).
    #   Z ego 1-hop: {Y} -> 1/1 = 1.0
    #   Z ego 2-hop: {Y} + neighbours of Y minus Z = {Y, X} -> 2/2 = 1.0
    #   Y ego 1-hop at step 2: {X} -> 1.0; 2-hop {X, Z}? Z has step 2 = same
    #   step, edge Y->Z visible at 2 -> 1-hop {X, Z} -> 2/2 = 1.0
    x, y, z = "330000001", "330000002", "330000003"
    data_dir = _ingest_and_run(
        monkeypatch, tmp_path, [x, y, z], [1, 2, 3], ["1", "1", "2"], [(x, y), (y, z)]
    )
    feats = _features(data_dir)
    assert feats[z][2] == 0.0  # no reverse edge
    assert feats[z][3] == 1.0
    assert feats[z][4] == 1.0
    assert feats[y][3] == 1.0
    assert feats[y][4] == 1.0


# --- P3-04: Louvain community ---------------------------------------------------


_TWO_TRIANGLE_IDS = [f"34000000{i}" for i in range(1, 7)]


def _two_triangle_setup() -> tuple[list[str], list[int], list[str], list[tuple[str, str]]]:
    # Two disconnected triangles {a1,a2,a3}, {b1,b2,b3}; hand-known partition.
    ids = _TWO_TRIANGLE_IDS
    a1, a2, a3, b1, b2, b3 = ids
    edges = [(a1, a2), (a2, a3), (a3, a1), (b1, b2), (b2, b3), (b3, b1)]
    steps = [1, 2, 3, 1, 2, 3]
    labels = ["1", "1", "2", "2", "unknown", "2"]
    return ids, steps, labels, edges


def test_louvain_two_cluster_partition_deterministic(tmp_path, monkeypatch) -> None:
    ids, steps, labels, edges = _two_triangle_setup()
    data_dir = _ingest_and_run(monkeypatch, tmp_path, ids, steps, labels, edges)
    feats = _features(data_dir)
    a1, a2, a3, b1, b2, b3 = ids
    assert feats[a1][5] == feats[a2][5] == feats[a3][5]
    assert feats[b1][5] == feats[b2][5] == feats[b3][5]
    assert feats[a1][5] != feats[b1][5]

    # Re-run: fixed seed -> identical partition.
    result = runner.invoke(app, ["graph-features", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    assert _features(data_dir) == feats


# --- P3-05: completeness gate fail-closed ---------------------------------------


def test_completeness_gate_row_shortfall_fails_closed(
    tmp_path, monkeypatch
) -> None:
    import aml_workbench.graph as graph_mod

    ids, steps, labels, edges = _two_triangle_setup()
    data_dir = _ingest_and_run(monkeypatch, tmp_path, ids, steps, labels, edges)

    original = graph_mod._compute_features

    def _lossy(edges_, steps_, labels_):
        features = original(edges_, steps_, labels_)
        features.pop(next(iter(features)))  # drop one tx -> incomplete coverage
        return features

    monkeypatch.setattr(graph_mod, "_compute_features", _lossy)
    result = runner.invoke(app, ["graph-features", "--data-dir", str(data_dir)])
    assert result.exit_code == 1
    assert "coverage is incomplete" in result.output


def test_full_frozen_extract_hand_check(tmp_path, monkeypatch) -> None:
    # Two triangles bridged by x(step 4, licit): a3->x, x->b3.
    # Hand-computed (visibility = later endpoint step):
    #   a1(step 1): frozen on empty graph  -> all-zero degrees, recip 0, tsa 0
    #   a2(step 2): in 1 (a1 illicit)      -> ego1 1/1 = 1.0, tsa 2-1 = 1
    #   a3(step 3): in 1 out 1, no reverse -> recip 0; ego1 {a2,a1} both
    #               illicit -> 1.0; tsa 3-2 = 1
    #   b2(step 2): in 1 (b1 licit)        -> ego1 0/1 = 0.0, tsa 1
    #   b3(step 3): in 1 out 1, no reverse -> recip 0; ego1 {b2 unknown, b1
    #               licit} -> 0/2 = 0.0; tsa 1
    #   x(step 4):  in 1 (a3) out 1 (b3), no reverse -> recip 0;
    #               ego1 {a3 licit, b3 licit} -> 0/2 = 0.0;
    #               ego2 adds {a1,a2,b1,b2} -> illicit 2/6; tsa 4-3 = 1
    ids = [f"35000000{i}" for i in range(1, 8)]
    a1, a2, a3, b1, b2, b3, x = ids
    steps = [1, 2, 3, 1, 2, 3, 4]
    labels = ["1", "1", "2", "2", "unknown", "2", "2"]
    edges = [
        (a1, a2),
        (a2, a3),
        (a3, a1),
        (b1, b2),
        (b2, b3),
        (b3, b1),
        (a3, x),
        (x, b3),
    ]
    data_dir = _ingest_and_run(monkeypatch, tmp_path, ids, steps, labels, edges)
    feats = _features(data_dir)

    assert feats[a1][:2] == (0, 0)
    assert feats[a2][:2] == (1, 0)
    assert feats[a2][3] == 1.0
    assert feats[a2][6] == 1
    assert feats[a3][:2] == (1, 1)
    assert feats[a3][2] == 0.0
    assert feats[a3][3] == 1.0
    assert feats[a3][6] == 1
    assert feats[b2][:2] == (1, 0)
    assert feats[b2][3] == 0.0
    assert feats[b2][6] == 1
    assert feats[b3][:2] == (1, 1)
    assert feats[b3][2] == 0.0
    assert feats[b3][3] == 0.0
    assert feats[b3][6] == 1
    assert feats[x][:2] == (1, 1)
    assert feats[x][2] == 0.0
    assert feats[x][3] == 0.0
    assert feats[x][4] == 2 / 6
    assert feats[x][6] == 1
    # No NULLs anywhere in required features.
    assert all(all(v is not None for v in row) for row in feats.values())
