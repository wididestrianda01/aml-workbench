"""Graph features (Track A): NetworkX over the Elliptic temporal graph.

Writes the `tx_graph_features` model table — one row per transaction, keyed by
txId VARCHAR — consumed identically by baselines, challenger, and the GNN.

Strict inductive protocol: a transaction at time step t only ever sees edges
whose *later* endpoint has step <= t. No future adjacency is reachable from any
row's feature vector, so train rows (steps 1-34) and test rows (35-49) are
symmetric under "graph known up to my own step" — no test-period adjacency can
leak into training-time features.

Feature set (spec Phase 3):
- in_degree / out_degree at the tx's own step
- reciprocity: fraction of incident edges whose reverse edge exists
- ego illicit fraction 1-hop / 2-hop: illicit neighbours over ALL neighbours
  (unknown-class nodes are graph context and stay in the denominator)
- louvain_community: Louvain id on the train-period graph (steps 1-34), fixed
  seed; -1 when the tx has no train-period presence (categorical downstream)
- time_since_activity: tx step minus the most recent strictly-earlier
  neighbour step; 0 when no earlier neighbour exists
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import networkx as nx

from aml_workbench import config
from aml_workbench.errors import DataQualityError

_FEATURE_COLUMNS = (
    ("in_degree", "INTEGER"),
    ("out_degree", "INTEGER"),
    ("reciprocity", "DOUBLE"),
    ("ego_illicit_1hop", "DOUBLE"),
    ("ego_illicit_2hop", "DOUBLE"),
    ("louvain_community", "INTEGER"),
    ("time_since_activity", "SMALLINT"),
)


def _open_graph_db(data_dir: Path) -> duckdb.DuckDBPyConnection:
    db_path = data_dir / "workbench.duckdb"
    if not db_path.exists():
        raise DataQualityError(
            f"workbench database not found at {db_path}; run `aml ingest` first"
        )
    con = duckdb.connect(str(db_path), read_only=True)
    tables = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    required = {"elliptic_tx", "elliptic_tx_features", "elliptic_edge"}
    if not required <= tables:
        con.close()
        raise DataQualityError(
            f"DuckDB store {db_path} lacks Elliptic ingest tables {sorted(required)}; "
            "run `aml ingest --track elliptic` first."
        )
    return con


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Fetch a single integer scalar (gate count queries)."""
    rows = con.execute(sql).fetchall()
    return int(rows[0][0])


def _illicit_fraction(neighbours: set[str], labels: dict[str, int | None]) -> float:
    """Illicit share of a neighbourhood; unknowns stay in the denominator."""
    if not neighbours:
        return 0.0
    illicit = sum(1 for n in neighbours if labels.get(n) == 1)
    return illicit / len(neighbours)


def _node_features(
    graph: nx.DiGraph,
    tx: str,
    steps: dict[str, int],
    labels: dict[str, int | None],
) -> tuple[int, int, float, float, float, int]:
    preds = set(graph.predecessors(tx)) if graph.has_node(tx) else set()
    succs = set(graph.successors(tx)) if graph.has_node(tx) else set()
    in_degree, out_degree = len(preds), len(succs)

    # Reciprocity: each incident edge counts once; reciprocated if the reverse
    # edge exists (s->u counts when u->s exists, and vice versa).
    incident = in_degree + out_degree
    reciprocated = sum(1 for s in preds if graph.has_edge(tx, s)) + sum(
        1 for d in succs if graph.has_edge(d, tx)
    )
    reciprocity = reciprocated / incident if incident else 0.0

    neighbours_1 = preds | succs
    neighbours_2 = set()
    for n in neighbours_1:
        neighbours_2 |= set(graph.predecessors(n)) | set(graph.successors(n))
    neighbours_2 -= neighbours_1
    neighbours_2.discard(tx)

    earlier = [steps[n] for n in neighbours_1 if steps[n] < steps[tx]]
    time_since_activity = steps[tx] - max(earlier) if earlier else 0

    return (
        in_degree,
        out_degree,
        reciprocity,
        _illicit_fraction(neighbours_1, labels),
        _illicit_fraction(neighbours_1 | neighbours_2, labels),
        time_since_activity,
    )


def _compute_features(
    edges: list[tuple[str, str]],
    steps: dict[str, int],
    labels: dict[str, int | None],
) -> dict[str, tuple[int, int, float, float, float, int, int]]:
    """Per-tx features with a per-tx point-in-time cutoff.

    Edges are grouped by their later endpoint's step; at step t the graph
    grows by exactly the edges that become visible at t, then the txs of
    step t are frozen against that graph state.
    """
    for src, dst in edges:
        if src not in steps or dst not in steps:
            raise DataQualityError(
                f"Edge ({src}, {dst}) references a tx absent from "
                "elliptic_tx_features; ingest gate violated."
            )

    txs_by_step: dict[int, list[str]] = {}
    for tx, step in steps.items():
        txs_by_step.setdefault(step, []).append(tx)
    edges_by_visible_step: dict[int, list[tuple[str, str]]] = {}
    for src, dst in edges:
        edges_by_visible_step.setdefault(max(steps[src], steps[dst]), []).append(
            (src, dst)
        )

    # Louvain pre-pass: partition over the train-period graph (all edges
    # visible by TRAIN_STEP_MAX), fixed seed; nodes outside it get -1.
    train_graph = nx.DiGraph()
    for step in sorted(edges_by_visible_step):
        if step > config.TRAIN_STEP_MAX:
            break
        for src, dst in edges_by_visible_step[step]:
            train_graph.add_edge(src, dst)
    communities: dict[str, int] = {}
    if train_graph.number_of_edges():
        partition = nx.community.louvain_communities(
            train_graph.to_undirected(), seed=config.GRAPH_SEED
        )
        communities = {
            node: cid for cid, nodes in enumerate(partition) for node in nodes
        }

    graph = nx.DiGraph()
    features: dict[str, tuple[int, int, float, float, float, int, int]] = {}
    for step in sorted(set(txs_by_step) | set(edges_by_visible_step)):
        for src, dst in edges_by_visible_step.get(step, ()):
            graph.add_edge(src, dst)
        for tx in txs_by_step.get(step, ()):
            in_deg, out_deg, recip, e1, e2, tsa = _node_features(
                graph, tx, steps, labels
            )
            features[tx] = (in_deg, out_deg, recip, e1, e2, communities.get(tx, -1), tsa)
    return features


def run_graph_features(data_dir: Path) -> str:
    """Compute graph features and write `tx_graph_features`. Fail-closed:
    missing tables, dangling edges, row-count mismatch, or NULLs in required
    features raise DataQualityError before the command reports success."""
    con = _open_graph_db(data_dir)
    try:
        edges = [
            (str(src), str(dst))
            for src, dst in con.execute(
                "SELECT src_tx_id, dst_tx_id FROM elliptic_edge "
                "ORDER BY src_tx_id, dst_tx_id"
            ).fetchall()
        ]
        steps = {
            str(tx): int(step)
            for tx, step in con.execute(
                "SELECT tx_id, time_step FROM elliptic_tx_features"
            ).fetchall()
        }
        labels = {
            str(tx): (int(label) if label is not None else None)
            for tx, label in con.execute(
                "SELECT tx_id, class_label FROM elliptic_tx"
            ).fetchall()
        }
    finally:
        con.close()

    features = _compute_features(edges, steps, labels)

    columns = ", ".join(f"{name} {sql_type}" for name, sql_type in _FEATURE_COLUMNS)
    placeholders = ", ".join("?" for _ in _FEATURE_COLUMNS)
    db_path = data_dir / "workbench.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE tx_graph_features (tx_id VARCHAR, {columns})"
        )
        con.executemany(
            f"INSERT INTO tx_graph_features VALUES (?, {placeholders})",
            [(tx, *values) for tx, values in sorted(features.items())],
        )

        # Completeness gate: exactly one row per tx, no NULLs, tx sets equal.
        n_rows = _scalar(con, "SELECT count(*) FROM tx_graph_features")
        n_tx = _scalar(con, "SELECT count(*) FROM elliptic_tx_features")
        if n_rows != n_tx:
            raise DataQualityError(
                f"tx_graph_features has {n_rows} rows for {n_tx} txs in "
                "elliptic_tx_features; feature coverage is incomplete."
            )
        missing = _scalar(
            con,
            "SELECT count(*) FROM ("
            "  (SELECT tx_id FROM elliptic_tx_features EXCEPT"
            "   SELECT tx_id FROM tx_graph_features)"
            "  UNION ALL"
            "  (SELECT tx_id FROM tx_graph_features EXCEPT"
            "   SELECT tx_id FROM elliptic_tx_features)"
            ")",
        )
        if missing != 0:
            raise DataQualityError(
                f"{missing} tx ids mismatch between tx_graph_features and "
                "elliptic_tx_features."
            )
        null_check = " OR ".join(f"{name} IS NULL" for name, _ in _FEATURE_COLUMNS)
        nulls = _scalar(
            con, f"SELECT count(*) FROM tx_graph_features WHERE {null_check}"
        )
        if nulls != 0:
            raise DataQualityError(
                f"{nulls} rows in tx_graph_features carry NULLs in required "
                "features."
            )
    finally:
        con.close()

    n_train = sum(1 for tx in features if steps[tx] <= config.TRAIN_STEP_MAX)
    return (
        f"Graph features: {len(features)} tx rows written to tx_graph_features "
        f"({len(edges)} edges; communities from steps 1-{config.TRAIN_STEP_MAX}; "
        f"{n_train} train-side rows)"
    )
