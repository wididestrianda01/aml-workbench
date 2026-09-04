"""Model training: raw-feature baselines and the calibrated LightGBM challenger.

Locked protocol: temporal split only (train steps 1-34, test 35-49), never
random; labeled rows only; PR-AUC is the predeclared challenger metric and the
promote-or-retain decision is recorded as an artifact, never asserted in prose.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from aml_workbench import config, db
from aml_workbench.errors import AmlWorkbenchError, DataQualityError
from aml_workbench.graph import FEATURE_COLUMNS

# Names of the graph-feature model table columns, derived once from the graph
# stage's own definition so adding a feature cannot drift apart.
GRAPH_FEATURE_COLUMNS = tuple(name for name, _ in FEATURE_COLUMNS)


@dataclass(frozen=True)
class RunMetrics:
    """One model, one seed, one temporal-split evaluation."""

    model: str
    seed: int
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class ChallengerResult:
    metrics: list[RunMetrics]
    best_seed: int
    mean_pr_auc: float
    val_pr_aucs: list[float] = field(default_factory=list)
    ablation: RunMetrics | None = None

def load_labeled(
    db_path: Path,
    *,
    include_graph: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], list[str]]:
    """Labeled rows only: X (n, d), y (illicit=1), steps — ordered by tx_id.

    ``include_graph`` appends the graph-feature model table columns to the raw
    features. Fail-closed: missing ingest (or feature) tables raise
    DataQualityError.
    """
    feature_cols = ", ".join(f"f{i:03d}" for i in range(1, config.FEATURE_COUNT + 1))
    graph_join = ""
    graph_cols = ""
    required = {
        "elliptic_tx": "run `aml ingest` first",
        "elliptic_tx_features": "run `aml ingest` first",
    }
    if include_graph:
        required["tx_graph_features"] = "run `aml graph-features` first"
        graph_cols = ", " + ", ".join(f"g.{name}" for name in GRAPH_FEATURE_COLUMNS)
        graph_join = "JOIN tx_graph_features g USING (tx_id)"
    query = f"""
        SELECT f.time_step, t.class_label, {feature_cols}{graph_cols}
        FROM elliptic_tx_features f
        JOIN elliptic_tx t USING (tx_id)
        {graph_join}
        WHERE t.class_label IS NOT NULL
        ORDER BY f.tx_id
    """
    con = db.open_workbench(db_path.parent, required, read_only=True)
    try:
        rows = con.execute(query).fetchnumpy()
    finally:
        con.close()

    steps = np.asarray(rows["time_step"], dtype=np.int64)
    class_label = np.asarray(rows["class_label"], dtype=np.int64)
    y = (class_label == 1).astype(np.int64)
    names = [f"f{i:03d}" for i in range(1, config.FEATURE_COUNT + 1)]
    if include_graph:
        names += GRAPH_FEATURE_COLUMNS
    x = np.column_stack([np.asarray(rows[name], dtype=np.float64) for name in names])
    if np.isnan(x).any():
        raise DataQualityError("NULL/NaN feature values found in the labeled training set.")
    return x, y, steps, names


def split_temporal(
    steps: NDArray[np.int64], protocol: Literal["test", "tuning"] = "test"
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Locked temporal split — the one seam every model stage goes through.

    protocol="test": train steps 1-TRAIN_STEP_MAX, test TEST_STEP_MIN-49.
    Never random. protocol="tuning": train 1-TUNING_TRAIN_STEP_MAX,
    validation TUNING_VAL_STEP_MIN-MAX — a selection slice carved strictly
    from training steps; the test side never enters selection.
    """
    if protocol == "tuning":
        train_mask = steps <= config.TUNING_TRAIN_STEP_MAX
        second_mask = (
            steps >= config.TUNING_VAL_STEP_MIN
        ) & (steps <= config.TUNING_VAL_STEP_MAX)
    else:
        train_mask = steps <= config.TRAIN_STEP_MAX
        second_mask = steps >= config.TEST_STEP_MIN
    if not train_mask.any() or not second_mask.any():
        raise DataQualityError(
            f"{protocol} temporal split produced an empty train or test side."
        )
    return train_mask, second_mask


def _mean_pr_auc(metrics: list[RunMetrics], model: str, attr: str = "pr_auc") -> float:
    values = [getattr(m, attr) for m in metrics if m.model == model]
    return float(np.mean(values))


def _render_baselines_report(
    metrics: list[RunMetrics], models: tuple[str, ...], seeds: tuple[int, ...]
) -> str:
    lines = [
        "# Baselines Report",
        "",
        f"LR/RF on {config.FEATURE_COUNT} raw features. Temporal split only, "
        f"never random: train steps 1-{config.TRAIN_STEP_MAX}, test steps "
        f"{config.TEST_STEP_MIN}-49. Seeds: {', '.join(map(str, seeds))}.",
        "",
    ]
    for m in metrics:
        lines.append(f"| {m.model} | {m.seed} | {m.roc_auc:.4f} | {m.pr_auc:.4f} |")
    for name in models:
        lines.append(
            f"| {name} | mean | {_mean_pr_auc(metrics, name, attr='roc_auc'):.4f} | "
            f"{_mean_pr_auc(metrics, name):.4f} |"
        )
    lines += [
        "",
        "PR-AUC is the predeclared challenger metric: the challenger must beat the "
        "best baseline mean PR-AUC to be promoted.",
        "",
    ]
    return "\n".join(lines)


def fit_lr_rf(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    train_mask: NDArray[np.bool_],
    test_mask: NDArray[np.bool_],
    *,
    seed: int,
    log: Callable[[RunMetrics], None] | None = None,
) -> list[RunMetrics]:
    """Standardize on train stats only, then fit/score the LR/RF candidate
    pair. Shared by the baselines stage and the smoke gate so the smoke gate
    exercises exactly the models it gates."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_test = scaler.transform(x[test_mask])
    y_train, y_test = y[train_mask], y[test_mask]
    candidates: tuple[tuple[str, Any], ...] = (
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    )
    metrics: list[RunMetrics] = []
    for name, model in candidates:
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_test)[:, 1]
        metrics.append(
            RunMetrics(
                model=name,
                seed=seed,
                roc_auc=float(roc_auc_score(y_test, scores)),
                pr_auc=float(average_precision_score(y_test, scores)),
            )
        )
        if log is not None:
            log(metrics[-1])
    return metrics


def run_baselines(data_dir: Path) -> str:
    """Train LR/RF raw-feature baselines across the configured seeds; write the
    comparison report and machine-readable metrics. Fail-closed on missing tables."""
    x, y, steps, _ = load_labeled(db.path(data_dir))
    train_mask, test_mask = split_temporal(steps)
    metrics: list[RunMetrics] = []
    for seed in config.MODEL_SEEDS:
        for m in fit_lr_rf(x, y, train_mask, test_mask, seed=seed):
            metrics.append(m)
            _log_run(
                data_dir,
                f"baselines_{m.model}",
                {"model": m.model, "seed": seed, "features": "raw"},
                {"roc_auc": m.roc_auc, "pr_auc": m.pr_auc},
            )

    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "baselines_report.md"
    report.write_text(
        _render_baselines_report(
            metrics, ("logistic_regression", "random_forest"), config.MODEL_SEEDS
        ),
        encoding="utf-8",
    )
    mean_pr_auc = {
        name: _mean_pr_auc(metrics, name) for name in ("logistic_regression", "random_forest")
    }
    payload: dict[str, object] = {
        "seeds": list(config.MODEL_SEEDS),
        "metrics": [m.__dict__ for m in metrics],
        "mean_pr_auc": mean_pr_auc,
    }
    (report_dir / "baselines_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return (
        "baselines: "
        + ", ".join(f"{name} mean PR-AUC {mean_pr_auc[name]:.4f}" for name in mean_pr_auc)
        + f" -> {report}"
    )


def _log_run(
    data_dir: Path,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log one training run to the local MLflow sqlite store under the data root."""
    mlflow.set_tracking_uri(f"sqlite:///{data_dir / config.MLFLOW_DB_NAME}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def run_tuning(data_dir: Path) -> dict[str, Any]:
    """Deterministic grid search over the LightGBM hyperparameters with early
    stopping on a validation slice carved strictly from training steps
    (train 1-30 / validate 31-34). The test side never enters selection.
    Returns the winning parameters; writes the tuning report."""
    x, y, steps, _ = load_labeled(db.path(data_dir), include_graph=True)
    train_mask, val_mask = split_temporal(steps, protocol="tuning")

    keys = list(config.LGBM_GRID)
    trials: list[tuple[float, dict[str, float | int]]] = []
    for values in itertools.product(*(config.LGBM_GRID[k] for k in keys)):
        params: dict[str, Any] = dict(zip(keys, values, strict=True))
        booster = lgb.LGBMClassifier(
            n_estimators=config.LGBM_N_ESTIMATORS,
            class_weight="balanced",
            random_state=config.MODEL_SEEDS[0],
            n_jobs=-1,
            verbose=-1,
            bagging_fraction=0.8,
            bagging_freq=1,
            **params,
        )
        booster.fit(
            x[train_mask],
            y[train_mask],
            eval_set=[(x[val_mask], y[val_mask])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(config.TUNING_EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        scores: dict[str, dict[str, float]] = booster.best_score_
        val_pr_auc = float(scores["valid_0"]["average_precision"])
        trials.append((val_pr_auc, params))

    # deterministic winner: best validation PR-AUC, ties broken by parameter order
    trials.sort(key=lambda t: (-t[0], str(sorted(t[1].items()))))
    best_val, best_params = trials[0]
    # bagging is fixed outside the grid but must ride along with the winner so
    # run_challenger trains with it (and random_state actually matters)
    best_params = {**best_params, "bagging_fraction": 0.8, "bagging_freq": 1}
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LightGBM Tuning Report",
        "",
        f"Grid search, {len(trials)} trials. Selection on validation steps "
        f"{config.TUNING_VAL_STEP_MIN}-{config.TUNING_VAL_STEP_MAX} only "
        f"(early stopping, patience {config.TUNING_EARLY_STOPPING_ROUNDS}); "
        f"the test side (steps {config.TEST_STEP_MIN}-49) never enters selection.",
        "",
        "| " + " | ".join(config.LGBM_GRID) + " | val PR-AUC |",
        "|---" * (len(config.LGBM_GRID) + 1) + "|",
    ]
    for val, p in trials:
        cells = " | ".join(str(p[k]) for k in config.LGBM_GRID)
        lines.append(f"| {cells} | {val:.4f} |")
    (report_dir / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "tuning_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    _log_run(
        data_dir,
        "challenger_tuning",
        {"trials": len(trials), **best_params},
        {"val_pr_auc": best_val},
    )
    return best_params


def run_challenger(data_dir: Path, params: dict[str, Any] | None = None) -> ChallengerResult:
    """Train the class-weighted, calibrated LightGBM challenger (raw + graph
    features) across the configured seeds; persist the best-seed model and the
    challenger metrics. Fail-closed on missing tables."""
    tuned = params is None
    if params is None:
        params = {
            "num_leaves": config.LGBM_NUM_LEAVES,
            "learning_rate": config.LGBM_LEARNING_RATE,
            "min_child_samples": 20,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
        }
    x, y, steps, feature_names = load_labeled(db.path(data_dir), include_graph=True)
    train_mask, test_mask = split_temporal(steps)
    # validation slice carved from training steps only — seed selection never
    # touches the test side
    _, val_mask = split_temporal(steps, protocol="tuning")
    x_train, x_test, x_val = x[train_mask], x[test_mask], x[val_mask]
    y_train, y_test, y_val = y[train_mask], y[test_mask], y[val_mask]
    metrics: list[RunMetrics] = []
    fitted: dict[int, lgb.LGBMClassifier] = {}
    val_pr_aucs: list[float] = []
    for seed in config.MODEL_SEEDS:
        base = lgb.LGBMClassifier(
            n_estimators=config.LGBM_N_ESTIMATORS,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        base.fit(x_train, y_train)  # keeps the persisted booster usable by the explainer
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(x_train, y_train)
        val_scores = calibrated.predict_proba(x_val)[:, 1]
        val_pr_aucs.append(float(average_precision_score(y_val, val_scores)))
        scores = calibrated.predict_proba(x_test)[:, 1]
        test_metrics = RunMetrics(
            model="lightgbm_challenger",
            seed=seed,
            roc_auc=float(roc_auc_score(y_test, scores)),
            pr_auc=float(average_precision_score(y_test, scores)),
        )
        metrics.append(test_metrics)
        fitted[seed] = base  # uncalibrated booster: the tree explainer needs raw trees
        _log_run(
            data_dir,
            "challenger_training",
            {"seed": seed, "n_estimators": config.LGBM_N_ESTIMATORS, **params},
            {
                "val_pr_auc": val_pr_aucs[-1],
                "roc_auc": test_metrics.roc_auc,
                "pr_auc": test_metrics.pr_auc,
            },
        )
    # best seed by validation-slice PR-AUC, ties broken by seed order; the
    # validation rows sit inside the training window, so no test leak. The
    # slice is in-sample for the booster — accepted ladder ceiling, recorded
    # beside the test numbers so the choice is auditable.
    best_seed = max(
        zip(config.MODEL_SEEDS, val_pr_aucs, strict=True), key=lambda t: (t[1], -t[0])
    )[0]


    # Deployment-honest ablation: retrain WITHOUT the neighbor-label features
    # (ego_illicit_1hop/2hop), which assume illicit labels of in-window
    # neighbours are known at scoring time. Records the production-realistic
    # floor beside the headline number.
    keep = [i for i, n in enumerate(feature_names) if not n.startswith("ego_illicit")]
    base_abl = lgb.LGBMClassifier(
        n_estimators=config.LGBM_N_ESTIMATORS,
        class_weight="balanced",
        random_state=best_seed,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    base_abl.fit(x_train[:, keep], y_train)
    calib_abl = CalibratedClassifierCV(base_abl, method="isotonic", cv=3)
    calib_abl.fit(x_train[:, keep], y_train)
    abl_scores = calib_abl.predict_proba(x_test[:, keep])[:, 1]
    ablation = RunMetrics(
        model="lightgbm_challenger_no_ego_labels",
        seed=best_seed,
        roc_auc=float(roc_auc_score(y_test, abl_scores)),
        pr_auc=float(average_precision_score(y_test, abl_scores)),
    )

    result = ChallengerResult(
        metrics=metrics,
        best_seed=best_seed,
        mean_pr_auc=_mean_pr_auc(metrics, "lightgbm_challenger"),
        val_pr_aucs=val_pr_aucs,
        ablation=ablation,
    )

    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": fitted[best_seed], "feature_names": feature_names, "seed": best_seed},
        models_dir / "challenger.joblib",
    )
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "challenger_metrics.json").write_text(
        json.dumps(
            {
                "seeds": list(config.MODEL_SEEDS),
                "best_seed": best_seed,
                "mean_pr_auc": result.mean_pr_auc,
                "metrics": [m.__dict__ for m in metrics],
                "val_pr_auc_by_seed": val_pr_aucs,
                "seed_effect": (
                    "bagging_fraction 0.8 / bagging_freq 1 makes random_state "
                    "operative — per-seed metrics are now independent draws"
                ),
                "ablation": result.ablation.__dict__ if result.ablation else None,
                "ablation_note": (
                    "retrained without ego_illicit_1hop/2hop: deployment-honest "
                    "floor (neighbour illicit labels are unavailable at scoring "
                    "time in production)"
                ),
                "seed_selection": (
                    f"best PR-AUC on validation steps "
                    f"{config.TUNING_VAL_STEP_MIN}-{config.TUNING_VAL_STEP_MAX} "
                    "(training steps only); test metrics feed the decision, "
                    "never the selection"
                ),
                "class_weight": "balanced",
                "calibration": {"method": "isotonic", "cv": 3},
                "parameters": params,
                "hyperparameter_source": "defaults" if tuned else "tuned",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def decide_promotion(data_dir: Path, challenger: ChallengerResult) -> Path:
    """Predeclared PR-AUC decision: promote only if the challenger mean PR-AUC
    beats the best baseline mean PR-AUC by the configured minimum gain.
    Fail-closed: missing baseline metrics raise instead of deciding from memory."""
    baselines_path = db.require(
        db.report_path(data_dir, "baselines_metrics.json"),
        "run 'aml baselines' before deciding.",
    )
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
    best_model = max(baselines["mean_pr_auc"], key=lambda k: baselines["mean_pr_auc"][k])
    best_baseline = float(baselines["mean_pr_auc"][best_model])
    gain = challenger.mean_pr_auc - best_baseline
    verdict = "promote" if gain >= config.CHALLENGER_MIN_PR_AUC_GAIN else "retain"
    if challenger.ablation is None:
        raise AmlWorkbenchError(
            "challenger result lacks the deployment-honest ablation; rerun `aml challenger`"
        )
    decision_path = db.report_path(data_dir, "decision_report.md")
    decision_path.write_text(
        "\n".join(
            [
                "# Challenger Decision",
                "",
                f"- Verdict: **{verdict}**",
                f"- Challenger mean PR-AUC ({challenger.mean_pr_auc:.4f}, best seed "
                f"{challenger.best_seed}, seeds {', '.join(map(str, config.MODEL_SEEDS))})",
                f"- Best baseline: {best_model} at {best_baseline:.4f} mean PR-AUC",
                f"- Gain: {gain:+.4f} (minimum to promote: "
                f"{config.CHALLENGER_MIN_PR_AUC_GAIN:+.4f})",
                f"- Deployment-honest ablation (no ego_illicit_1hop/2hop): PR-AUC "
                f"{challenger.ablation.pr_auc:.4f}, ROC-AUC "
                f"{challenger.ablation.roc_auc:.4f} — the headline assumes "
                "in-window neighbour illicit labels are known at scoring time; "
                "the ablation is the production-realistic floor.",
                f"- Caveat: validation-slice PR-AUC is in-sample for the booster "
                f"({', '.join(f'{v:.4f}' for v in challenger.val_pr_aucs)} across "
                "seeds); a perfect 1.0 there means the validation slice is "
                "trivially separable and seed selection carries no signal — "
                "the test-side numbers above are the decision basis.",
                "",
                "Predeclared metric: PR-AUC on the temporal-split test side "
                f"(steps {config.TEST_STEP_MIN}-49). Class-weighted, isotonic-calibrated "
                "LightGBM over raw + graph features.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return decision_path
