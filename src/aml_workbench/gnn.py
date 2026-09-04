"""Strict-inductive GraphSAGE baseline (PyG) and the honest GNN-vs-GBM report.

Locked protocol: temporal split only (train steps 1-34, test 35-49), never
random; unknown-class nodes are graph context, never train or score targets.
Strict induction is enforced at the edge level: the message-passing edge set
keeps only edges whose LATER endpoint sits at or before the training boundary,
so no test-period adjacency enters training or scoring. An explicit assertion
verifies the mask rather than trusting it; a leaky edge set fails the run
closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from aml_workbench import config, db
from aml_workbench.errors import DataQualityError
from aml_workbench.model import split_temporal

_PROTOCOL_NOTE = (
    "Strict-inductive GraphSAGE (PyG): message-passing edges restricted to "
    "edges whose later endpoint is at or before the training boundary "
    f"(step {config.TRAIN_STEP_MAX}); no test-period adjacency enters training "
    "or scoring; unknown-class nodes are graph context only. Protocol "
    "positioning: transductive GNN protocols that let test-period adjacency "
    "into message passing inflate GNN metrics by a large margin (Maganti 2026 "
    "protocol audit; Weber et al. 2019) — this baseline refuses that artifact, "
    "so an honest loss or tie against the GBM is the expected, reportable "
    "finding, not a failure."
)


def _load_graph(
    db_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """All-node feature matrix, labels (illicit=1, licit=0, unknown=-1), steps,
    and edge index with each edge's later-endpoint step. Node order is fixed by
    tx_id so features and edges index the same node array. Fail-closed."""
    feature_cols = ", ".join(f"f{i:03d}" for i in range(1, config.FEATURE_COUNT + 1))
    con = db.open_workbench(
        db_path.parent,
        {
            "elliptic_tx": "run `aml ingest` first",
            "elliptic_tx_features": "run `aml ingest` first",
            "elliptic_edge": "run `aml ingest` first",
        },
        read_only=True,
    )
    try:
        nodes = con.execute(
            f"""
            SELECT f.tx_id, f.time_step, t.class_label, {feature_cols}
            FROM elliptic_tx_features f
            JOIN elliptic_tx t USING (tx_id)
            ORDER BY f.tx_id
            """
        ).fetchnumpy()
        edges = con.execute(
            """
            SELECT e.src_tx_id, e.dst_tx_id,
                   s1.time_step AS src_step, s2.time_step AS dst_step
            FROM elliptic_edge e
            JOIN elliptic_tx_features s1 ON s1.tx_id = e.src_tx_id
            JOIN elliptic_tx_features s2 ON s2.tx_id = e.dst_tx_id
            """
        ).fetchnumpy()
    finally:
        con.close()

    n = len(nodes["tx_id"])
    x = np.column_stack(
        [
            np.asarray(nodes[f"f{i:03d}"], dtype=np.float64)
            for i in range(1, config.FEATURE_COUNT + 1)
        ]
    )
    if np.isnan(x).any():
        raise DataQualityError("NULL/NaN feature values found in the graph node set.")
    class_label = np.asarray(nodes["class_label"], dtype=np.float64)
    y = np.where(class_label == 1, 1, np.where(class_label == 2, 0, -1)).astype(np.int64)
    steps = np.asarray(nodes["time_step"], dtype=np.int64)
    tx_index = {tx: i for i, tx in enumerate(nodes["tx_id"])}
    src = np.asarray([tx_index[t] for t in edges["src_tx_id"]], dtype=np.int64)
    dst = np.asarray([tx_index[t] for t in edges["dst_tx_id"]], dtype=np.int64)
    edge_later_step = np.maximum(
        np.asarray(edges["src_step"], dtype=np.int64),
        np.asarray(edges["dst_step"], dtype=np.int64),
    )
    if len(src) != len(edges["src_tx_id"]) or n == 0:
        raise DataQualityError("Graph load produced an empty node or edge set.")
    del tx_index
    return x, y, steps, np.vstack([src, dst, edge_later_step])


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.out(self.conv2(h, edge_index)).squeeze(-1)  # type: ignore[no-any-return]


def _inductive_edge_mask(edge_later_step: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Edges whose later endpoint is at or before the training boundary."""
    return edge_later_step <= config.TRAIN_STEP_MAX


def _assert_no_lookahead(edge_index: NDArray[np.int64], steps: NDArray[np.int64]) -> None:
    """Split-boundary assertion: no retained edge touches a node beyond the
    training boundary. The mask is verified, not trusted — a leak fails closed."""
    later = np.maximum(steps[edge_index[0]], steps[edge_index[1]])
    if later.size and later.max() > config.TRAIN_STEP_MAX:
        raise DataQualityError(
            f"look-ahead detected: message-passing edge crosses step {config.TRAIN_STEP_MAX}"
        )


def _train_one(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    steps: NDArray[np.int64],
    edge_index: NDArray[np.int64],
    seed: int,
) -> dict[str, Any]:
    """One seed: standardize on train-node stats, train GraphSAGE, score the
    labeled test side. Returns per-seed metrics."""
    torch.manual_seed(seed)

    labeled = (y == 1) | (y == 0)
    train_mask, test_mask = split_temporal(steps)
    train_mask &= labeled
    test_mask &= labeled
    if not train_mask.any() or not test_mask.any():
        raise DataQualityError("Temporal split produced an empty train or test side.")
    _assert_no_lookahead(edge_index, steps)

    x_t = torch.from_numpy(x)
    # standardize on TRAIN-node stats only; test rows never set scale
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    x_t = ((x_t - torch.from_numpy(mean)) / torch.from_numpy(std)).float()

    data = Data(
        x=x_t,
        edge_index=torch.from_numpy(edge_index.astype(np.int64)),
    )
    y_t = torch.from_numpy(y.astype(np.float32))
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))

    model = GraphSAGE(config.FEATURE_COUNT, config.GNN_HIDDEN)
    pos_weight = torch.tensor(float((y[train_mask] == 0).sum()) / float((y[train_mask] == 1).sum()))
    optim = torch.optim.Adam(model.parameters(), lr=config.GNN_LR)
    model.train()
    for _ in range(config.GNN_EPOCHS):
        optim.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], y_t[train_idx], pos_weight=pos_weight
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optim.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(data.x, data.edge_index)).numpy()
    y_test = y[test_mask]
    return {
        "seed": seed,
        "roc_auc": float(roc_auc_score(y_test, scores[test_mask])),
        "pr_auc": float(average_precision_score(y_test, scores[test_mask])),
    }


def run_gnn(data_dir: Path) -> str:
    """Train the strict-inductive GraphSAGE baseline across the configured
    seeds; write gnn_metrics.json and the honest GNN-vs-GBM comparison.
    Fail-closed on missing tables or a missing challenger artifact."""
    x, y, steps, edges = _load_graph(db.path(data_dir))
    keep = _inductive_edge_mask(edges[2])
    edge_index = edges[:2, keep]

    per_seed = [_train_one(x, y, steps, edge_index, seed) for seed in config.MODEL_SEEDS]
    pr = [m["pr_auc"] for m in per_seed]
    roc = [m["roc_auc"] for m in per_seed]

    # fail-closed ordering: validate the challenger artifact BEFORE any
    # artifact is written, so a violation leaves no partial output
    db.require(
        db.report_path(data_dir, "challenger_metrics.json"),
        "run `aml challenger` before `aml gnn`.",
    )
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "graphsage_strict_inductive",
        "protocol": _PROTOCOL_NOTE,
        "seeds": list(config.MODEL_SEEDS),
        "per_seed": per_seed,
        "mean_roc_auc": float(np.mean(roc)),
        "mean_pr_auc": float(np.mean(pr)),
        "std_roc_auc": float(np.std(roc)),
        "std_pr_auc": float(np.std(pr)),
    }
    (report_dir / "gnn_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = (
        f"gnn: GraphSAGE strict-inductive, {len(per_seed)} seeds, "
        f"mean ROC-AUC {payload['mean_roc_auc']:.4f}, mean PR-AUC {payload['mean_pr_auc']:.4f}"
    )
    return summary + "; " + _write_comparison(data_dir, payload)


def _write_comparison(data_dir: Path, gnn: dict[str, Any]) -> str:
    """Honest GNN-vs-GBM comparison: mean±std per model, explicit loss/tie
    verdict, protocol positioning. Fail-closed on a missing challenger artifact."""
    challenger_path = db.require(
        db.report_path(data_dir, "challenger_metrics.json"),
        "run `aml challenger` before `aml gnn`.",
    )
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    challenger = json.loads(challenger_path.read_text(encoding="utf-8"))
    cb_pr = [float(m["pr_auc"]) for m in challenger["metrics"]]
    cb_roc = [float(m["roc_auc"]) for m in challenger["metrics"]]

    delta = float(np.mean(cb_pr)) - gnn["mean_pr_auc"]
    if delta > 0.01:
        verdict = "GNN loses to the GBM"
    elif delta < -0.01:
        verdict = "GNN beats the GBM"
    else:
        verdict = "tie"
    lines = [
        "# GNN vs GBM Comparison",
        "",
        "| model | mean ROC-AUC | mean PR-AUC |",
        "|---|---|---|",
        f"| GraphSAGE (strict-inductive) | {gnn['mean_roc_auc']:.4f} ± {gnn['std_roc_auc']:.4f} "
        f"| {gnn['mean_pr_auc']:.4f} ± {gnn['std_pr_auc']:.4f} |",
        f"| LightGBM challenger | {float(np.mean(cb_roc)):.4f} ± {float(np.std(cb_roc)):.4f} "
        f"| {float(np.mean(cb_pr)):.4f} ± {float(np.std(cb_pr)):.4f} |",
        "",
        f"**Verdict: {verdict}** (PR-AUC delta {delta:+.4f}; 3 seeds each: "
        f"{', '.join(map(str, config.MODEL_SEEDS))}).",
        "",
        _PROTOCOL_NOTE,
        "",
    ]
    (report_dir / "gnn_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "gnn_comparison.json").write_text(
        json.dumps(
            {
                "gnn": {
                    "mean_roc_auc": gnn["mean_roc_auc"],
                    "mean_pr_auc": gnn["mean_pr_auc"],
                    "std_roc_auc": gnn["std_roc_auc"],
                    "std_pr_auc": gnn["std_pr_auc"],
                    "seeds": gnn["seeds"],
                },
                "challenger": {
                    "mean_roc_auc": float(np.mean(cb_roc)),
                    "mean_pr_auc": float(np.mean(cb_pr)),
                    "std_roc_auc": float(np.std(cb_roc)),
                    "std_pr_auc": float(np.std(cb_pr)),
                    "seeds": challenger["seeds"],
                },
                "pr_auc_delta": delta,
                "verdict": verdict,
                "protocol": _PROTOCOL_NOTE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return f"verdict {verdict} (PR-AUC delta {delta:+.4f}) -> {report_dir / 'gnn_comparison.md'}"
