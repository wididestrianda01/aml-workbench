"""Report artifacts. The one-page smoke report is rendered here;
later stages extend this module with validation reports and the technical report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aml_workbench import __version__

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
