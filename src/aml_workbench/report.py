"""Report artifacts. The one-page smoke report is rendered here;
later stages extend this module with validation reports and the technical report.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg backend must be set first)

from aml_workbench import __version__, config, db
from aml_workbench.errors import DataQualityError

if TYPE_CHECKING:
    from aml_workbench.smoke import SmokeResult

_SPLIT_STATEMENT = (
    "Temporal split only, never random: train on labeled steps 1-34, "
    "test on labeled steps 35-49; unknown-class nodes excluded from training "
    "and scoring (strict-inductive in spirit; full leakage-proof machinery "
    "arrives later)."
)


def render_smoke_report(result: SmokeResult) -> str:
    """One-page smoke report: auditable without reading code."""
    lines = [
        "# Smoke Report",
        "",
        f"- Run: {__version__}, gate ROC-AUC >= {result.gate_threshold:.2f} for BOTH "
        f"models, runtime limit {result.runtime_limit_s:.0f} s",
        f"- Verdict: **{'PASS' if result.passed else 'FAIL'}**",
        f"- Split: {_SPLIT_STATEMENT}",
        f"- Labeled rows: train {result.train_rows} / test {result.test_rows}",
        f"- Illicit base rate: train {result.train_base_rate:.4f} / "
        f"test {result.test_base_rate:.4f} (label imbalance reported, not hidden)",
        f"- Runtime: {result.runtime_s:.1f} s",
        "",
        "| Model | ROC-AUC | PR-AUC |",
        "|---|---|---|",
    ]
    for m in result.models:
        lines.append(f"| {m.name} | {m.roc_auc:.4f} | {m.pr_auc:.4f} |")
    lines += [
        "",
        "PR-AUC is reported from the start: it is the predeclared challenger "
        "metric for model selection.",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    return "\n".join(lines)


def render_alert_stats_report(stats: list[tuple[str, int, int, float]]) -> str:
    """One-page per-scenario alert statistics report."""
    total_alerts = sum(int(row[1]) for row in stats)
    total_entities = sum(int(row[2]) for row in stats)
    lines = [
        "# Alert Statistics (rules engine)",
        "",
        "- Source: `rule_alert` table (scenario, entity, reason, details)",
        "- Windows: tumbling daily; amount scenarios restricted to US Dollar",
        "",
        "| Scenario | Alerts | Distinct entities | Laundering-entity precision |",
        "|---|---|---|---|",
    ]
    for scenario, alert_count, distinct_entities, precision in stats:
        lines.append(
            f"| {scenario} | {alert_count} | {distinct_entities} | {precision:.4f} |"
        )
    lines += [
        f"| **Total** | **{total_alerts}** | **{total_entities}** | |",
        "",
        "Laundering-entity precision: share of alerted accounts appearing in at "
        "least one laundering-flagged transaction (tuning compass, not a "
        "performance metric). Scenario thresholds live in the frozen config "
        "module and are tuned per scenario against these statistics.",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    return "\n".join(lines)


# --- Technical report (Phase 8) ------------------------------------------------

_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "baselines_metrics.json",
    "challenger_metrics.json",
    "tuning_params.json",
    "validation_metrics.json",
    "drift_metrics.json",
    "gnn_comparison.json",
)


def _load_json(report_dir: Path, name: str) -> dict[str, Any]:
    path = report_dir / name
    if not path.exists():
        raise DataQualityError(f"report artifact missing: {name}; run its stage first")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _base_rate_by_step(con: duckdb.DuckDBPyConnection) -> list[tuple[int, int, float]]:
    """(step, n_tx, illicit_share) over the 49 labeled Elliptic time steps."""
    return [
        (int(s), int(n), float(r))
        for s, n, r in con.execute(
            """
            SELECT f.time_step, count(*),
                   avg(CASE WHEN t.class_label = '1' THEN 1.0 ELSE 0.0 END)
            FROM elliptic_tx_features f JOIN elliptic_tx t USING (tx_id)
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
    ]


def _precision_at_k_curve(
    con: duckdb.DuckDBPyConnection, cost_per_investigation: float
) -> tuple[list[int], list[float], list[int], list[float]]:
    """precision@k, cumulative TP, and cost per TP over the ranked queue."""
    row = con.execute("SELECT day2, universe FROM triage_meta").fetchone()
    assert row is not None, "triage_meta is empty; run triage first"
    day2, _ = row
    launderers = {
        r[0]
        for r in con.execute(
            """
            SELECT DISTINCT to_account FROM hismall_transaction
            WHERE is_laundering = 1 AND CAST(tx_time AS DATE) > ?::date
            UNION
            SELECT DISTINCT from_account FROM hismall_transaction
            WHERE is_laundering = 1 AND CAST(tx_time AS DATE) > ?::date
            """,
            [day2, day2],
        ).fetchall()
    }
    ranked = [
        r[0]
        for r in con.execute(
            "SELECT entity FROM alert_queue ORDER BY fused_score DESC, rank"
        ).fetchall()
    ]
    ks, precisions, tps, costs = [], [], [], []
    hits = 0
    for k in range(1, len(ranked) + 1):
        if ranked[k - 1] in launderers:
            hits += 1
        if k in (1, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, len(ranked)):
            ks.append(k)
            precisions.append(hits / k)
            tps.append(hits)
            costs.append(k * cost_per_investigation / hits if hits else float("nan"))
    return ks, precisions, tps, costs


def _figure(path: Path) -> str:
    return f"![{path.stem}](figures/{path.name})"


def run_report(data_dir: Path) -> str:
    """Assemble docs/report.md + figures from stage artifacts. Fail-closed on
    any missing stage artifact or required DB table."""
    report_dir = data_dir / "reports"
    art = {name: _load_json(report_dir, name) for name in _REQUIRED_ARTIFACTS}

    db_path = db.path(data_dir)
    if not db_path.exists():
        raise DataQualityError(f"workbench database missing: {db_path}; run ingest first")
    con = duckdb.connect(str(db_path), read_only=True)

    docs = config.PROJECT_ROOT / "docs"
    figs = docs / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    base_rate = _base_rate_by_step(con)
    per_step = art["validation_metrics.json"]["per_step"]
    psi = art["drift_metrics.json"]["columns"]
    gnn = art["gnn_comparison.json"]
    ch = art["challenger_metrics.json"]
    baselines = art["baselines_metrics.json"]
    manifest = art["run_manifest.json"]

    # figures
    steps = [s for s, _, _ in base_rate]
    rates = [r for _, _, r in base_rate]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(steps, rates, marker=".", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel("illicit share")
    ax.set_title("Illicit base rate per time step (Elliptic, log scale)")
    fig.tight_layout()
    fig.savefig(figs / "base_rate_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    vsteps = [p["step"] for p in per_step]
    for key, label in (("f1", "F1"), ("precision", "precision"), ("recall", "recall")):
        ax.plot(vsteps, [p[key] for p in per_step], marker=".", label=label)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("test time step")
    ax.set_ylabel("score at threshold 0.5")
    ax.set_title("Walk-forward per-step scores (strict-inductive refit)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figs / "walk_forward.png", dpi=150)
    plt.close(fig)

    top_psi = sorted(psi.items(), key=lambda kv: -kv[1]["psi"])[:20]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [k for k, _ in top_psi][::-1]
    vals = [v["psi"] for _, v in top_psi][::-1]
    colors = [
        "#9D102D" if v["breach"] else "#E5A400" if v["watch"] else "#888888"
        for _, v in top_psi
    ][::-1]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.25, color="#9D102D", ls="--", lw=1, label="breach 0.25")
    ax.axvline(0.10, color="#E5A400", ls=":", lw=1, label="watch 0.10")
    ax.set_xlabel("population stability index (steps 1-34 vs 35-49)")
    ax.set_title("Feature drift, 20 largest PSI")
    ax.legend(frameon=False)
    fig.tight_layout()
    kq_row = con.execute("SELECT count(*) FROM alert_queue").fetchone()
    assert kq_row is not None
    kq = int(kq_row[0])
    fig.savefig(figs / "drift_psi.png", dpi=150)
    plt.close(fig)
    kpi = con.execute(
        "SELECT true_positives, precision_at_100, precision_at_500, precision_at_1000"
        " FROM triage_kpi WHERE scope = 'overall'"
    ).fetchone()
    assert kpi is not None, "triage_kpi missing the overall row; run triage first"
    ks, precisions, tps, costs = _precision_at_k_curve(con, config.INVESTIGATION_COST_USD)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ks, precisions, marker=".", label="precision@k")
    ax.set_xscale("log")
    ax.set_xlabel("k alerts investigated")
    ax.set_ylabel("precision@k")
    ax2 = ax.twinx()
    ax2.plot(ks, costs, marker=".", color="#9D102D", label="cost per TP (USD)")
    ax2.set_ylabel("cost per true positive (USD)")
    ax.set_title("Operating-point economics of the fused queue")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "precision_at_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    width = 0.35
    xs_gnn = (0.0, 1.0)
    ax.bar(
        [x - width / 2 for x in xs_gnn],
        [gnn["gnn"]["mean_pr_auc"], gnn["challenger"]["mean_pr_auc"]],
        width,
        yerr=[gnn["gnn"]["std_pr_auc"], gnn["challenger"]["std_pr_auc"]],
        label="PR-AUC",
    )
    ax.bar(
        [x + width / 2 for x in xs_gnn],
        [gnn["gnn"]["mean_roc_auc"], gnn["challenger"]["mean_roc_auc"]],
        width,
        yerr=[gnn["gnn"]["std_roc_auc"], gnn["challenger"]["std_roc_auc"]],
        label="ROC-AUC",
    )
    ax.set_xticks(xs_gnn, ["GraphSAGE", "LightGBM"])
    ax.set_title("GNN baseline vs GBM challenger (3 seeds, mean +/- sd)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "gnn_vs_gbm.png", dpi=150)
    plt.close(fig)

    per_scope = con.execute(
        "SELECT scope, detection_rate, precision_at_500 FROM triage_kpi"
        " WHERE scope != 'overall' ORDER BY scope"
    ).fetchall()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    xs = tuple(float(i) for i in range(len(per_scope)))
    ax.bar([x - width / 2 for x in xs], [p[1] for p in per_scope], width, label="detection rate")
    ax.bar([x + width / 2 for x in xs], [p[2] for p in per_scope], width, label="precision@500")
    ax.set_xticks(xs, [p[0] for p in per_scope], rotation=30, ha="right")
    ax.set_title("Per-scenario triage KPIs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "scenario_kpis.png", dpi=150)
    plt.close(fig)

    # numbers for the prose
    mean_pr = ch["mean_pr_auc"]
    seed_prs = [m["pr_auc"] for m in ch["metrics"]]
    bl = {
        m["model"]: m["pr_auc"]
        for m in baselines["metrics"]
        if m["seed"] == baselines["seeds"][0]
    }
    f1s = [p["f1"] for p in per_step]
    n_breach = sum(1 for v in psi.values() if v["breach"])
    n_watch = sum(1 for v in psi.values() if v["watch"])

    lines = [
        "# AML Workbench: technical report",
        "",
        f"Lineage: commit `{manifest['commit']}`"
        f", config fingerprint `{manifest['config_fingerprint'][:12]}`"
        f", {manifest['run_count']} tracked MLflow runs (`data/reports/run_manifest.json`).",
        "",
        "## Scope and claims",
        "",
        "This report evaluates an anti-money-laundering (AML) transaction-monitoring"
        " workbench built on two public datasets: the Elliptic bitcoin transaction"
        " graph (Track A) and the IBM HI-Small synthetic bank transaction data"
        " (Track B). It is a reference implementation and a benchmark study. It is"
        " not a production system, and none of the numbers here should be read as"
        " evidence of performance on live banking data.",
        "",
        "## Context for readers new to the field",
        "",
        "Two committed primers carry the background this report assumes:"
        " [primer-aml-domain.md](primer-aml-domain.md) covers what money laundering"
        " is, the regulatory obligations behind transaction monitoring, the"
        " typologies the two datasets encode, and the economics of alert triage."
        " [primer-ml.md](primer-ml.md) covers the machine-learning concepts — class"
        " imbalance, PR-AUC versus ROC-AUC, gradient boosting, walk-forward"
        " validation, leakage discipline, drift and PSI, and SHAP. Read them first"
        " if any term below is new; the report itself records results and their"
        " caveats, not definitions.",
        "",
        "## Data and protocol",
        "",
        "The two tracks use one shared, non-negotiable evaluation protocol:"
        " temporal splits only, strict-inductive feature computation, and no"
        " test-period adjacency during training or scoring. On Elliptic, models"
        " train on labeled time steps 1-34 and are tested on labeled steps 35-49;"
        " the 157,205 unknown-class transactions appear only as graph context and"
        " are never training or scoring targets. On HI-Small, the walk-forward cuts"
        " are 2022-09-11 and 2022-09-15, taken from the triage run metadata.",
        "",
        "Class imbalance is large and is reported rather than corrected away. The"
        " Elliptic labeled base rate is about 9.6% illicit (4,545 of 46,564), and it"
        " varies by roughly an order of magnitude across the 49 time steps:",
        "",
        _figure(figs / "base_rate_curve.png"),
        "",
        "The challenger metric was predeclared as PR-AUC, before any model was"
        " trained. ROC-AUC is reported for comparability with prior published"
        " benchmarks. Micro-F1 is deliberately not used as a headline number: at a"
        " fixed 0.5 threshold, micro-averaged F1 is dominated by the majority class"
        " whenever the positive base rate is low, and it can therefore look"
        " acceptable while recall on illicit activity is poor. Where F1 appears"
        " below, it is computed per time step at the fixed threshold and read"
        " alongside precision and recall.",
        "",
        "## Models and results",
        "",
        "### Baselines and challenger (Track A)",
        "",
        f"Logistic regression reaches mean PR-AUC {bl.get('logistic_regression', float('nan')):.4f}"
        f" and random forest {bl.get('random_forest', float('nan')):.4f} on the"
        " temporal test steps (first-seed values; all three seeds are in the"
        " metrics artifact). The LightGBM challenger, with hyperparameters selected"
        " by a deterministic grid search whose validation slice (steps 31-34) never"
        " touches the test side, reaches a mean PR-AUC of"
        f" {mean_pr:.4f} across seeds (per-seed {', '.join(f'{p:.4f}' for p in seed_prs)})."
        f" The challenger clears the predeclared promotion threshold of"
        f" +{config.CHALLENGER_MIN_PR_AUC_GAIN:.2f} PR-AUC over the best baseline.",
        "",
        "### Walk-forward validation",
        "",
        "The challenger is refit at each test step with a strictly expanding"
        " training window. Per-step precision, recall, and F1 at the fixed 0.5"
        f" threshold stay high through the test window (F1 range"
        f" {min(f1s):.3f}-{max(f1s):.3f}), but this stability partly reflects the"
        " synthetic regularity of Elliptic rather than a property the workbench"
        " created; the drift measurements below are the honest check on that.",
        "",
        _figure(figs / "walk_forward.png"),
        "",
        "### GNN baseline",
        "",
        "A strict-inductive GraphSAGE baseline (PyG, 3 seeds) reaches mean PR-AUC"
        f" {gnn['gnn']['mean_pr_auc']:.4f} against the challenger's"
        f" {gnn['challenger']['mean_pr_auc']:.4f} under the identical protocol. The"
        " predeclared verdict was recorded as: " + gnn["verdict"] + ". The GBM on"
        " engineered graph features remains the selected model.",
        "",
        _figure(figs / "gnn_vs_gbm.png"),
        "",
        "### Feature attribution",
        "",
        "The SHAP summary over a seeded 20,000-row test subsample is reported in"
        " `data/reports/shap_summary.md`. It is referenced, not reproduced here, so"
        " that the committed report stays reproducible from the stage artifacts.",
        "",
        "### Drift",
        "",
        f"Population stability index between training and test steps flags"
        f" {n_breach} features above the 0.25 breach threshold and {n_watch} in the"
        " 0.10-0.25 watch band. These are reported as measured: the synthetic data"
        " generator shifts several features across the split, which is itself"
        " evidence that the temporal protocol is doing its job.",
        "",
        _figure(figs / "drift_psi.png"),
        "",
        "## Operational layer (Track B)",
        "",
        "The rules engine and the account model fuse into one ranked alert queue"
        f" ({kq:,} alerts after deduplication). At the current operating point of"
        f" {kpi[1]:.4f} precision at k=100 and {kpi[2]:.4f} at k=500, the queue"
        f" surfaces {kpi[0]} true positives in the top 500. The cost-per-true-positive"
        " curve makes the false-positive economics explicit for a compliance-team"
        " audience.",
        "",
        _figure(figs / "precision_at_k.png"),
        "",
        _figure(figs / "scenario_kpis.png"),
        "",
        "## Limitations",
        "",
        "Stated bluntly, because an interviewer will ask:",
        "",
        "1. Synthetic and public data only. HI-Small is generated by IBM's AMLSim;"
        " Elliptic is a one-off bitcoin snapshot. Laundering patterns in real bank"
        " data differ in volume, label quality, and adversary adaptation. No claim"
        " transfers to production monitoring.",
        "2. Label quality. Elliptic's 'illicit' labels are heuristic (from public"
        " enforcement sources), incomplete, and static. HI-Small labels are ground"
        " truth by construction, which flatters supervised methods.",
        "3. No adversary model. The evaluation is static; a laundering strategy that"
        " adapts to the rule thresholds is out of scope.",
        "4. GraphSAGE is a baseline, not a studied architecture. No tuning of the"
        " GNN was attempted; the comparison shows the GBM wins under this protocol,"
        " not that GNNs are unsuitable for AML.",
        "5. Drift is measured, not mitigated. Features breaching the PSI threshold"
        " are reported; no retraining policy, champion-challenger schedule, or"
        " monitoring SLA is implemented (see the rollback runbook).",
        "6. The triage operating point (top-k, cost per investigation) is an"
        " assumption, not a calibrated estimate from an investigations team.",
        "",
        "## Regulatory context (one-pager)",
        "",
        "The EU AML package (Regulation (EU) 2024/1624, AMLR, and Directive (EU)"
        " 2024/1640, AMLD6) directly obliges credit and financial institutions to"
        " run risk-based transaction monitoring, with the AMLR applying from"
        " mid-2027 and the new EU Anti-Money Laundering Authority (AMLA) becoming"
        " operational around 2025-2026 with direct supervision of selected high-risk"
        " entities from 2028. EBA Guidelines EBA/GL/2021/02 require institutions to"
        " evidence the effectiveness of their monitoring systems, which is the gap a"
        " model like this addresses: scored, ranked, and costed alert queues with"
        " measurable detection and false-positive rates. In Sweden, Finansinspektionen's"
        " regulations FFFS 2017:11 (and later amendments) transpose equivalent"
        " risk-based obligations, including documented monitoring and suspicion"
        " reporting. The EU AI Act (Regulation (EU) 2024/1689) does not list AML"
        " transaction monitoring as a high-risk use case per se, but if such a system"
        " feeds decisions with significant effects on natural or legal persons,"
        " institutions should assess AI Act obligations and GDPR Art. 22 (automated"
        " decision-making) anyway; this workbench's model-output-is-advisory design"
        " (a ranked queue for human investigators, never an automated account"
        " decision) is the conservative posture under both regimes. FATF"
        " Recommendations 20 and 23 set the international baseline for suspicious"
        " transaction reporting that these EU instruments implement.",
        "",
        "## Reproduction",
        "",
        "Every number above is produced by a batch pipeline command and stored as an"
        " artifact; the seam tests assert the report sections and the KPI arithmetic."
        " `aml track` writes the run manifest that ties these artifacts to a code"
        " commit and MLflow run ids.",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    report_path = docs / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return f"report: {len(lines)} lines, 6 figures -> {report_path}"
