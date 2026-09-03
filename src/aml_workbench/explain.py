"""SHAP summary on the persisted challenger model (cut-order slot one).

Computes TreeExplainer attributions on a seeded sample of the TEST-period
features and writes a report ranking features by mean absolute attribution —
raw and graph features named. Dropping this stage leaves the promote-or-retain
decision complete: it documents the promoted model, it never gates it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import shap

from aml_workbench import config
from aml_workbench.errors import DataQualityError
from aml_workbench.model import load_labeled, split_temporal


def run_shap(data_dir: Path) -> str:
    """Write the SHAP summary artifact for the persisted challenger.
    Fail-closed: no persisted model raises before anything is written."""
    model_path = data_dir / "models" / "challenger.joblib"
    if not model_path.is_file():
        raise DataQualityError(
            f"{model_path} not found; run 'aml challenger' before explaining."
        )
    bundle: dict[str, Any] = joblib.load(model_path)
    model = bundle["model"]
    feature_names: list[str] = bundle["feature_names"]

    db_path = data_dir / "workbench.duckdb"
    x, _, steps, _ = load_labeled(db_path, include_graph=True)
    train_mask, test_mask = split_temporal(steps)
    x_test = x[test_mask]
    if x_test.shape[0] > config.SHAP_SAMPLE_ROWS:
        rng = np.random.default_rng(config.MODEL_SEEDS[0])
        x_test = x_test[rng.choice(x_test.shape[0], config.SHAP_SAMPLE_ROWS, replace=False)]

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_test)
    if isinstance(values, list):  # binary classifiers may return per-class arrays
        values = values[-1]
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(-mean_abs)

    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "shap_summary.md"
    graph_names = {
        name for name in feature_names if not name.startswith("f")
    }
    top_feature = feature_names[int(order[0])]
    lines = [
        "# SHAP Summary",
        "",
        f"TreeExplainer on the persisted challenger (best seed {bundle['seed']}); "
        f"{x_test.shape[0]} TEST-period rows "
        f"(steps {config.TEST_STEP_MIN}-49, seeded sample). Attributions rank the "
        "promoted model's inputs; they explain it, they do not justify it.",
        "",
        f"Graph features ranked in the top 10: "
        f"{sorted(set(feature_names[int(i)] for i in order[:10]) & graph_names) or 'none'}",
        "",
        "| Rank | Feature | Mean |SHAP| |",
        "|---|---|---|",
    ]
    for rank, idx in enumerate(order, start=1):
        lines.append(f"| {rank} | {feature_names[int(idx)]} | {mean_abs[idx]:.4f} |")
    lines += [
        "",
        f"Top feature: {top_feature}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return f"shap: top feature {top_feature} -> {report_path}"
