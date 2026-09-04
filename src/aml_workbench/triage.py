"""Alert triage: fused rule+ML queue and operational KPIs, exposed through the
`aml triage` pipeline command.

Design decisions (documented, not hidden):
- **Walk-forward task.** Account behavior before a cutoff predicts laundering
  involvement after it. Cut points sit at fixed fractions of the observed
  timeline: train features @ cut1 -> labels in (cut1, cut2]; test features @
  cut2 -> labels after cut2. Never random.
- **Strict-inductive alert scoring.** Each alert is scored from the account's
  transaction history strictly BEFORE the alert's own day (parsed from the
  scenario `details` window). An account whose alert is its first activity has
  no history: its ML component is NULL and the fused score falls back to the
  rule component alone — recorded, not silently substituted.
- **Fusion.** Fused score is the weighted average of the ML component and the
  rule component (the scenario's laundering-entity precision from
  `alert_scenario_stats`). Weights are frozen config; component columns stay
  visible in the queue for interpretability.
- **KPIs at an operating point.** Detection rate, FPR, alerts/day,
  precision@k, investigation yield and cost per true positive are computed
  per scenario and overall at the configured operating point (top-k alerts
  investigated).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from aml_workbench import config, db, rules
from aml_workbench.errors import DataQualityError

if TYPE_CHECKING:
    import duckdb

# Canonical feature column order — one shared definition for snapshots,
# training and alert scoring so the model matrix can never drift apart.
FEATURES: tuple[str, ...] = (
    "tx_in_count",
    "tx_out_count",
    "amount_in_total",
    "amount_out_total",
    "counterparty_in_count",
    "counterparty_out_count",
    "payment_format_count",
    "currency_count",
    "max_amount_out",
    "active_day_count",
    "usd_subthreshold_share",
)

# Aggregation expressions over a `leg` CTE with columns
# (acct, tx_time, amt_in, amt_out, cpty, payment_currency, payment_format).
_FEATURE_EXPRS = f"""
    count(*) FILTER (WHERE amt_in > 0) AS tx_in_count,
    count(*) FILTER (WHERE amt_out > 0) AS tx_out_count,
    sum(amt_in) AS amount_in_total,
    sum(amt_out) AS amount_out_total,
    count(DISTINCT cpty) FILTER (WHERE amt_in > 0) AS counterparty_in_count,
    count(DISTINCT cpty) FILTER (WHERE amt_out > 0) AS counterparty_out_count,
    count(DISTINCT payment_format) AS payment_format_count,
    count(DISTINCT payment_currency) AS currency_count,
    max(amt_out) AS max_amount_out,
    count(DISTINCT date_trunc('day', tx_time)) AS active_day_count,
    count(*) FILTER (
        WHERE payment_currency = 'US Dollar' AND amt_out > 0
          AND amt_out < {config.REPORTING_THRESHOLD_USD}
    ) / NULLIF(count(*) FILTER (
        WHERE payment_currency = 'US Dollar' AND amt_out > 0
    ), 0) AS usd_subthreshold_share
"""

_SNAPSHOT_SQL = f"""
CREATE OR REPLACE TEMP TABLE {{name}} AS
WITH leg AS (
    SELECT to_account AS acct, tx_time, amount_received AS amt_in,
           0.0 AS amt_out, from_account AS cpty,
           payment_currency, payment_format
    FROM hismall_transaction WHERE tx_time < TIMESTAMP '{{as_of}}'
    UNION ALL
    SELECT from_account, tx_time, 0.0, amount_paid, to_account,
           payment_currency, payment_format
    FROM hismall_transaction WHERE tx_time < TIMESTAMP '{{as_of}}'
)
SELECT acct, {_FEATURE_EXPRS}
FROM leg
GROUP BY acct
"""



def _laundering_accounts(con: duckdb.DuckDBPyConnection, lo: date, hi: date) -> set[str]:
    """Accounts on a laundering-flagged transaction in the (lo, hi] window,
    compared on calendar day so a cut-day transaction stays in its window."""
    rows = con.execute(
        """
        SELECT DISTINCT acct FROM (
            SELECT to_account AS acct FROM hismall_transaction
            WHERE is_laundering = 1
              AND date_trunc('day', tx_time) > ?::DATE
              AND date_trunc('day', tx_time) <= ?::DATE
            UNION
            SELECT from_account FROM hismall_transaction
            WHERE is_laundering = 1
              AND date_trunc('day', tx_time) > ?::DATE
              AND date_trunc('day', tx_time) <= ?::DATE
        )
        """,
        [lo, hi, lo, hi],
    ).fetchall()
    return {str(r[0]) for r in rows}


def _snapshot_df(con: duckdb.DuckDBPyConnection, name: str, as_of: date) -> pd.DataFrame:
    """Per-account behavior features from all transactions strictly before as_of."""
    con.execute(_SNAPSHOT_SQL.format(name=name, as_of=as_of))
    cols = ", ".join(FEATURES)
    return con.execute(f"SELECT acct, {cols} FROM {name}").fetchdf()


def _fit_walk_forward(
    con: duckdb.DuckDBPyConnection, day1: date, day2: date, last_day: date
) -> tuple[lgb.LGBMClassifier, float]:
    """Train the HI-Small account model on the walk-forward split; return the
    fitted model and its test-window PR-AUC."""
    snap1 = _snapshot_df(con, "triage_snap_train", day1)
    snap2 = _snapshot_df(con, "triage_snap_test", day2)
    la_train = _laundering_accounts(con, day1, day2)
    la_test = _laundering_accounts(con, day2, last_day)
    if not la_train:
        raise DataQualityError(
            "triage train window is empty of laundering activity: no positive "
            "labels between the train and test cut points — the timeline is "
            "too short for the configured cut fractions"
        )
    if not la_test:
        raise DataQualityError(
            "triage test window is empty of laundering activity after the "
            "test cut point — cannot measure the queue"
        )

    y1 = snap1["acct"].isin(la_train).to_numpy()
    if len(np.unique(y1)) < 2:
        raise DataQualityError(
            "triage train slice has a single class: no account active before "
            "the train cut point turns laundering in the train label window"
        )
    model = lgb.LGBMClassifier(
        n_estimators=config.LGBM_N_ESTIMATORS,
        learning_rate=config.LGBM_LEARNING_RATE,
        num_leaves=config.LGBM_NUM_LEAVES,
        class_weight="balanced",
        random_state=config.MODEL_SEEDS[0],
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(snap1[list(FEATURES)].fillna(0.0), y1)

    y2 = snap2["acct"].isin(la_test).to_numpy()
    if len(np.unique(y2)) < 2:
        raise DataQualityError(
            "triage test slice has a single class: PR-AUC undefined"
        )
    scores = np.asarray(model.predict_proba(snap2[list(FEATURES)].fillna(0.0)))[:, 1]
    pr_auc = float(average_precision_score(y2, scores))
    return model, pr_auc


def _score_alerts(
    con: duckdb.DuckDBPyConnection, model: lgb.LGBMClassifier
) -> pd.DataFrame:
    """Parse alert days, score every alert from strictly-prior history, fuse
    with the rule component, and rank. Returns the full queue frame."""
    alerts = con.execute(
        f"""
        SELECT scenario, entity, reason, details,
               TRY_CAST({rules.ALERT_DAY_SQL} AS DATE) AS alert_day
        FROM rule_alert
        """
    ).fetchdf()
    if alerts.empty:
        raise DataQualityError(
            "rule_alert is empty; nothing to triage — check scenario "
            "thresholds or the underlying data before building the queue"
        )
    if alerts["alert_day"].isna().any():
        raise DataQualityError(
            "rule_alert rows missing a parsable 'window=YYYY-MM-DD' in "
            "details; per-alert as-of scoring cannot proceed"
        )

    rule = con.execute(
        """
        SELECT scenario, laundering_entity_precision FROM alert_scenario_stats
        """
    ).fetchall()
    rule_map = {str(s): float(p) for s, p in rule}
    alerts["rule_component"] = alerts["scenario"].map(rule_map).fillna(0.0)

    # Per-alert features: only transactions strictly before the alert's day.
    con.register("alerts_df", alerts)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE pairs AS "
        "SELECT DISTINCT entity, alert_day FROM alerts_df"
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE alert_feats AS
        WITH leg AS (
            SELECT p.entity AS acct, p.alert_day, t.tx_time,
                   t.amount_received AS amt_in, 0.0 AS amt_out,
                   t.from_account AS cpty, t.payment_currency, t.payment_format
            FROM pairs p JOIN hismall_transaction t
              ON t.to_account = p.entity AND t.tx_time < p.alert_day
            UNION ALL
            SELECT p.entity, p.alert_day, t.tx_time, 0.0, t.amount_paid,
                   t.to_account, t.payment_currency, t.payment_format
            FROM pairs p JOIN hismall_transaction t
              ON t.from_account = p.entity AND t.tx_time < p.alert_day
        )
        SELECT acct, alert_day, {_FEATURE_EXPRS}
        FROM leg
        GROUP BY acct, alert_day
        """
    )
    feat_df = con.execute(
        f"SELECT acct, alert_day, {', '.join(FEATURES)} FROM alert_feats"
    ).fetchdf()

    merged = alerts.merge(
        feat_df.rename(columns={"acct": "entity"}), on=["entity", "alert_day"], how="left"
    )
    x = merged[list(FEATURES)]
    ml: np.ndarray = np.full(len(merged), np.nan)
    known = x.notna().any(axis=1).to_numpy()
    if known.any():
        ml[known] = np.asarray(model.predict_proba(x.loc[known].fillna(0.0)))[:, 1]
    merged["ml_component"] = ml

    w_ml, w_rule = config.FUSION_ML_WEIGHT, config.FUSION_RULE_WEIGHT
    fused = w_ml * merged["ml_component"].fillna(0.0) + w_rule * merged["rule_component"]
    has_ml = merged["ml_component"].notna()
    fused = fused.where(has_ml, merged["rule_component"])  # no history -> rule alone
    merged["fused_score"] = fused

    ranked = merged.sort_values(
        ["fused_score", "ml_component", "entity", "alert_day", "scenario"],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    # One investigation per account-day: multiple rule scenarios firing on the
    # same (entity, alert_day) collapse to the highest-scoring row so KPI cost
    # math never double-counts investigations.
    ranked = ranked.drop_duplicates(subset=["entity", "alert_day"], keep="first")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked[
        [
            "rank",
            "scenario",
            "entity",
            "alert_day",
            "rule_component",
            "ml_component",
            "fused_score",
            "reason",
            "details",
        ]
    ]


def _kpi_rows(
    queue: pd.DataFrame,
    la: set[str],
    universe: int,
    days_eval: int,
    k_op: int | None = None,
    cost_per_investigation: float | None = None,
) -> list[dict[str, object]]:
    """Operational KPIs per scope (overall + per scenario) at the given
    operating point (default: config). All arithmetic here is deliberately
    plain so it can be hand-checked in tests."""
    k = k_op if k_op is not None else config.TRIAGE_OPERATING_K
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", queue)]
    scopes += [(s, queue[queue["scenario"] == s]) for s in sorted(queue["scenario"].unique())]
    for scope, scoped in scopes:
        ordered = scoped.sort_values(
            ["fused_score", "ml_component", "entity", "alert_day"],
            ascending=[False, False, True, True],
            na_position="last",
        )
        op = ordered.head(min(k, len(ordered)))
        entities = set(op["entity"].astype(str))
        tp = len(entities & la)
        alerted = len(entities)
        investigated = len(op)
        row: dict[str, object] = {
            "scope": scope,
            "alerts_investigated": investigated,
            "true_positives": tp,
            "detection_rate": round(tp / len(la), 4) if la else None,
            "false_positive_rate": round((alerted - tp) / (universe - len(la)), 6)
            if universe > len(la)
            else None,
            "alerts_per_day": round(investigated / days_eval, 4) if days_eval else None,
            "investigation_yield": round(tp / investigated, 4) if investigated else None,
            "cost_per_true_positive": round(
                investigated * (cost_per_investigation or config.INVESTIGATION_COST_USD) / tp, 2
            )
            if tp
            else None,
        }
        for k in config.PRECISION_AT_K:
            topk = ordered.head(min(k, len(ordered)))
            tp_k = len(set(topk["entity"].astype(str)) & la)
            row[f"precision_at_{k}"] = round(tp_k / len(topk), 4) if len(topk) else None
        rows.append(row)
    return rows


def run_triage(data_dir: Path) -> str:
    """Build the fused alert queue and KPI table; write the triage report.
    Fail-closed: missing or empty inputs raise DataQualityError before any
    output is written."""
    con = db.open_workbench(
        data_dir,
        {
            "rule_alert": "run `aml rules` first",
            "alert_scenario_stats": "run `aml alert-stats` first",
            "hismall_transaction": "run `aml ingest --track hi-small` first",
        },
    )
    try:
        if db.scalar(con, "SELECT count(*) FROM rule_alert") == 0:
            raise DataQualityError(
                "rule_alert is empty; every scenario fired zero alerts — "
                "nothing to triage"
            )

        bounds = con.execute(
            "SELECT min(tx_time), max(tx_time) FROM hismall_transaction"
        ).fetchone()
        if bounds is None or bounds[0] is None or bounds[1] is None:
            raise DataQualityError("hismall_transaction has no rows to triage over")
        first_day, last_day = bounds[0].date(), bounds[1].date()
        span = (last_day - first_day).days
        day1 = first_day + timedelta(days=round(span * config.TRIAGE_TRAIN_CUT_PCT))
        day2 = first_day + timedelta(days=round(span * config.TRIAGE_TEST_CUT_PCT))
        if day2 >= last_day:
            raise DataQualityError(
                "triage test window is empty: the data span is too short for "
                f"the configured cut fractions (span {span} days)"
            )

        model, pr_auc = _fit_walk_forward(con, day1, day2, last_day)
        queue = _score_alerts(con, model)
        con.register("queue_df", queue)
        con.execute("CREATE OR REPLACE TABLE alert_queue AS SELECT * FROM queue_df")

        la = _laundering_accounts(con, day2, last_day)
        universe = db.scalar(
            con,
            """
            SELECT count(*) FROM (
                SELECT from_account AS acct FROM hismall_transaction
                UNION
                SELECT to_account FROM hismall_transaction
            )
            """,
        )
        days_eval = (last_day - day2).days
        kpis = _kpi_rows(queue, la, universe, days_eval)
        kpi_df = pd.DataFrame(kpis)
        con.register("kpi_df", kpi_df)
        con.execute("CREATE OR REPLACE TABLE triage_kpi AS SELECT * FROM kpi_df")
        con.execute(
            """
            CREATE OR REPLACE TABLE triage_meta AS
            SELECT ?::DATE AS day1, ?::DATE AS day2,
                   ? AS universe, ? AS days_eval
            """,
            [day1, day2, universe, days_eval],
        )
    finally:
        con.close()

    report_path = data_dir / "reports" / "triage.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_triage_report(queue, kpis, pr_auc, day1, day2, la, universe),
        encoding="utf-8",
    )
    return (
        f"Triage run: {len(queue)} alerts ranked; test-window PR-AUC "
        f"{pr_auc:.4f}; KPI table triage_kpi written for "
        f"{len(kpis)} scopes"
    )


def _render_triage_report(
    queue: pd.DataFrame,
    kpis: list[dict[str, object]],
    pr_auc: float,
    day1: date,
    day2: date,
    la: set[str],
    universe: int,
) -> str:
    from datetime import UTC, datetime

    head = queue.head(20)
    lines = [
        "# Alert Triage Report",
        "",
        f"- Queue: {len(queue)} alerts fused from rule components and the "
        f"HI-Small account model (weights ML {config.FUSION_ML_WEIGHT:.2f} / "
        f"rule {config.FUSION_RULE_WEIGHT:.2f})",
        f"- Model task: account behavior before a cut predicts laundering "
        f"involvement after it; test-window PR-AUC **{pr_auc:.4f}**",
        f"- Cut points: train labels ({day1}, {day2}], evaluation after {day2} "
        f"({len(la)} laundering accounts of {universe} total)",
        "- Alerts with no prior history carry no ML component; their fused "
        "score is the rule component alone (recorded, not imputed)",
        "- Queue deduplicated to one alert per (entity, day): the "
        "highest-scoring scenario row survives, so KPI cost math never "
        "double-counts investigations",
        "",
        "## Top of queue",
        "",
        "| Rank | Scenario | Entity | Day | Rule | ML | Fused |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in head.iterrows():
        ml = f"{r['ml_component']:.4f}" if pd.notna(r["ml_component"]) else "—"
        lines.append(
            f"| {int(r['rank'])} | {r['scenario']} | {r['entity']} | "
            f"{r['alert_day']} | {r['rule_component']:.4f} | {ml} | "
            f"{r['fused_score']:.4f} |"
        )
    lines += [
        "",
        "## Operational KPIs",
        "",
    ]
    metric_cols = [c for c in kpis[0] if c != "scope"]
    lines.append("| Scope | " + " | ".join(metric_cols) + " |")
    lines.append("|---" * (len(metric_cols) + 1) + "|")
    for row in kpis:
        vals: list[str] = [
            f"{v:.4f}" if isinstance(v := row[c], float) else str(v if v is not None else "—")
            for c in metric_cols
        ]
        lines.append(f"| {row['scope']} | " + " | ".join(vals) + " |")
    lines += [
        "",
        f"Operating point: top {config.TRIAGE_OPERATING_K} alerts investigated "
        f"at {config.INVESTIGATION_COST_USD:.0f} USD each. precision@k reported "
        f"for k in {list(config.PRECISION_AT_K)}.",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    return "\n".join(lines)
