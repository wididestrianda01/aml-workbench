"""Report artifacts. Phase 1 renders the one-page smoke report (C5 evidence);
later phases extend this module with validation reports and the technical report.
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
    "arrives in Phase 5)."
)


def render_smoke_report(result: SmokeResult) -> str:
    """One-page smoke report: the Phase-1 evidence artifact, auditable without
    reading code."""
    lines = [
        "# AML Workbench Smoke Report (C5 gate)",
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
        "metric for Phase 4 model selection.",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    return "\n".join(lines)
