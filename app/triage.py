"""Thin Streamlit triage view — read-only investigator console over the
`aml triage` artifacts (alert_queue, triage_kpi, triage_meta).

Design decisions (documented, not hidden):
- The view computes nothing heavy: ranking, fusion and the model live in the
  pipeline; the only view-side computation is the operating-point cutoff over
  the stored fused scores, with KPIs recomputed from the shared arithmetic in
  `aml_workbench.triage._kpi_rows` (one formula, two surfaces).
- Cost-per-investigation is editable so the false-positive economics respond
  live — the KPI story investigators actually argue about.
- KTH brand palette on a flat, editorial surface: no gradients, no shadows,
  1px hairline borders, typographic hierarchy, monospace for meta values.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from aml_workbench import config
from aml_workbench.triage import _kpi_rows

# KTH profile colors (intra.kth.se graphic profile).
KTH_BLUE = "#1954A6"
KTH_LIGHTBLUE = "#24A0D8"
KTH_RED = "#9D102D"
KTH_GREEN = "#62922E"
KTH_DARKGRAY = "#65656C"
KTH_LIGHTGRAY = "#EAEAEA"
BONE = "#FBFBFA"
CHARCOAL = "#2F3437"

_CSS = f"""
<style>
    .stApp {{ background: {BONE}; color: {CHARCOAL}; }}
    section.main > div {{ padding-top: 1.5rem; }}
    h1 {{
        font-family: 'Helvetica Neue', sans-serif; letter-spacing: -0.03em;
        color: {CHARCOAL}; font-weight: 700;
    }}
    .kth-rule {{ border: 0; border-top: 3px solid {KTH_BLUE}; margin: 0 0 1rem 0; }}
    .kth-caption {{
        text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem;
        color: {KTH_DARKGRAY};
    }}
    .kth-card {{
        background: #FFFFFF; border: 1px solid {KTH_LIGHTGRAY};
        border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
    }}
    .kth-tag {{
        display: inline-block; border-radius: 9999px; padding: 0.1rem 0.6rem;
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em;
        background: #E1F3FE; color: #1F6C9F; margin-right: 0.4rem;
    }}
    .kth-meta {{ font-family: 'SF Mono', 'JetBrains Mono', monospace;
        font-size: 0.78rem; color: {KTH_DARKGRAY}; }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF; border: 1px solid {KTH_LIGHTGRAY};
        border-radius: 8px; padding: 0.75rem 1rem;
    div[data-testid="stMetricValue"] {{ color: {CHARCOAL} !important;
        font-weight: 600; }}
     div[data-testid="stMetricLabel"] p {{ color: {KTH_DARKGRAY} !important;
         font-size: 0.74rem !important; text-transform: uppercase;
         letter-spacing: 0.06em; }}
</style>
"""


@st.cache_data
def load_triage(data_dir: str) -> tuple[pd.DataFrame, str, str, int, int, set[str]]:
    """Load triage artifacts from a fresh read-only connection. Connections
    are not cached (they die between runs); the small data payload is."""
    path = Path(data_dir) / "workbench.duckdb"
    if not path.exists():
        st.error(f"workbench database not found at {path}; run `aml ingest` first.")
        st.stop()
    con = duckdb.connect(str(path), read_only=True)
    try:
        have = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        missing = [
            t for t in ("alert_queue", "triage_meta", "hismall_transaction") if t not in have
        ]
        if missing:
            st.error(
                "Missing triage artifacts: "
                + ", ".join(missing)
                + " — run `aml rules`, `aml alert-stats` and `aml triage` first."
            )
            st.stop()
        queue = con.execute("SELECT * FROM alert_queue").fetchdf()
        day1, day2, universe, days_eval = con.execute(
            "SELECT day1, day2, universe, days_eval FROM triage_meta"
        ).fetchone()
        la: set[str] = {
            str(r[0])
            for r in con.execute(
                """
                SELECT DISTINCT acct FROM (
                    SELECT to_account AS acct FROM hismall_transaction
                    WHERE is_laundering = 1 AND tx_time > ?::DATE
                    UNION
                    SELECT from_account FROM hismall_transaction
                    WHERE is_laundering = 1 AND tx_time > ?::DATE
                )
                """,
                [day2, day2],
            ).fetchall()
        }
    finally:
        con.close()
    return queue, str(day1), str(day2), int(universe), int(days_eval), la


def main() -> None:
    st.set_page_config(page_title="Alert Triage Console", page_icon="", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    data_dir = os.environ.get(config.DATA_DIR_ENV, str(config.default_data_dir()))
    queue, day1, day2, universe, days_eval, la = load_triage(data_dir)

    st.markdown("## Alert Triage Console")
    st.markdown(
        f'<div class="kth-caption">Fused rule + ML queue · evaluation window '
        f"after {day2} · cuts {day1} / {day2}</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="kth-rule">', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**Operating point**")
        scenario = st.selectbox(
            "Scenario", ["All scenarios", *sorted(queue["scenario"].unique())]
        )
        scoped = (
            queue if scenario == "All scenarios" else queue[queue["scenario"] == scenario]
        ).copy()
        k = st.slider(
            "Alerts investigated (top-k)", 1, max(len(scoped), 1),
            min(config.TRIAGE_OPERATING_K, max(len(scoped), 1)),
        )
        cost = st.slider(
            "Cost per investigation (USD)",
            min_value=10.0,
            max_value=500.0,
            value=config.INVESTIGATION_COST_USD,
            step=10.0,
        )

    # Shared KPI arithmetic at the view-side operating point; the cost slider
    # re-prices the queue: cost per TP = investigated * cost / TP.
    kpis = _kpi_rows(
        scoped, la, int(universe), int(days_eval), k_op=k, cost_per_investigation=cost
    )
    overall = kpis[0]

    def fmt(key: str) -> str:
        v = overall.get(key)
        if v is None:
            return "—"
        if key in ("alerts_per_day", "cost_per_true_positive"):
            return f"{v:,.1f}"
        return f"{v:.4f}"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Detection rate", fmt("detection_rate"))
    m2.metric("False-positive rate", fmt("false_positive_rate"))
    m3.metric("Alerts / day", fmt("alerts_per_day"))
    m4.metric("Precision @ k", fmt("investigation_yield"))
    m5.metric("True positives", str(overall["true_positives"]))
    m6.metric("Cost per TP (USD)", str(overall["cost_per_true_positive"] or "—"))

    st.markdown(
        f'<div class="kth-meta">investigating top {k} of {len(scoped)} alerts · '
        f"{len(la)} laundering accounts in the evaluation window · "
        f'{int(universe)} accounts total</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Ranked queue")
    show = scoped.head(50)[
        ["rank", "scenario", "entity", "alert_day", "rule_component", "ml_component", "fused_score"]
    ]
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "alert_day": st.column_config.DateColumn("Alert day", format="YYYY-MM-DD"),
            "rule_component": st.column_config.NumberColumn("Rule", format="%.4f"),
            "ml_component": st.column_config.NumberColumn("ML", format="%.4f"),
            "fused_score": st.column_config.NumberColumn("Fused", format="%.4f"),
        },
    )

    st.markdown("### Alert detail")
    for _, row in scoped.head(30).iterrows():
        ml = "—" if pd.isna(row["ml_component"]) else f"{row['ml_component']:.4f}"
        with st.expander(
            f"{int(row['rank'])}. {row['scenario']} · {row['entity']} · {row['fused_score']:.4f}"
        ):
            st.markdown(
                f'<span class="kth-tag">{row["scenario"]}</span>'
                f'<span class="kth-meta">rule {row["rule_component"]:.4f} · '
                f"ml {ml} · alert day {row['alert_day']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Reason:** {row['reason']}")
            st.markdown(f"```\n{row['details']}\n```")


main()
