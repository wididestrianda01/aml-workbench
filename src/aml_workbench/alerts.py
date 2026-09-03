"""Per-scenario alert statistics, exposed through the
`aml alert-stats` pipeline command.

Reads the `rule_alert` table written by `aml rules`, materializes per-scenario
counts into `alert_scenario_stats`, and renders the one-page alert-statistics
report artifact. Threshold tuning is per scenario exactly because these stats
exist.
"""

from __future__ import annotations

from pathlib import Path

from aml_workbench import db
from aml_workbench.errors import DataQualityError
from aml_workbench.report import render_alert_stats_report


def run_alert_stats(data_dir: Path) -> Path:
    """Compute per-scenario alert statistics and write the report artifact.
    Fail-closed: missing database or alert table raises DataQualityError."""
    con = db.open_workbench(data_dir, {"rule_alert": "run `aml rules` first"})
    try:
        if db.scalar(con, "SELECT count(*) FROM rule_alert") == 0:
            con.close()
            raise DataQualityError(
                "rule_alert is empty; every scenario fired zero alerts — check "
                "scenario thresholds or the underlying data before reporting"
            )
        con.execute(
            """
            CREATE OR REPLACE TABLE alert_scenario_stats AS
            WITH la AS (
                SELECT DISTINCT to_account AS acct FROM hismall_transaction
                WHERE is_laundering = 1
                UNION
                SELECT DISTINCT from_account FROM hismall_transaction
                WHERE is_laundering = 1
            ),
            per_entity AS (
                SELECT p.scenario, p.entity,
                       CASE WHEN la.acct IS NOT NULL THEN 1 ELSE 0 END AS hit
                FROM (SELECT DISTINCT scenario, entity FROM rule_alert) p
                LEFT JOIN la ON p.entity = la.acct
            ),
            totals AS (
                SELECT scenario, count(*) AS alert_count
                FROM rule_alert GROUP BY scenario
            )
            SELECT e.scenario,
                   any_value(t.alert_count) AS alert_count,
                   count(*) AS distinct_entities,
                   round(sum(e.hit) * 1.0 / count(*), 4) AS laundering_entity_precision
            FROM per_entity e
            JOIN totals t ON e.scenario = t.scenario
            GROUP BY e.scenario
            """
        )
        stats = con.execute(
            """
            SELECT scenario, alert_count, distinct_entities, laundering_entity_precision
            FROM alert_scenario_stats
            ORDER BY scenario
            """
        ).fetchall()
    finally:
        con.close()

    report_path = data_dir / "reports" / "alert_stats.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_alert_stats_report(stats), encoding="utf-8")
    return report_path
