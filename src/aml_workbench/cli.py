"""Pipeline CLI — the single test seam.

One typer subcommand per pipeline stage:
download -> ingest -> smoke -> rules -> alert-stats -> graph-features ->
baselines -> challenger -> shap -> validate -> drift -> gnn -> triage -> view ->
track -> report.

Fail-closed doctrine: unimplemented stages exit non-zero with a clear message;
gate violations exit non-zero before any downstream output is written.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from aml_workbench import config
from aml_workbench.errors import AmlWorkbenchError

app = typer.Typer(
    name="aml",
    help="AML workbench — batch pipeline commands (fail-closed).",
    no_args_is_help=True,
)

DataDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--data-dir",
        help="Data root (default: ./data or $AML_WORKBENCH_DATA_DIR).",
        show_default=False,
    ),
]


class Track(StrEnum):
    elliptic = "elliptic"
    hi_small = "hi-small"
    all = "all"


def _data_dir(value: Path | None) -> Path:
    return value if value is not None else config.default_data_dir()


def _not_implemented(stage: str) -> None:
    typer.echo(f"Fail-closed: stage '{stage}' is not implemented yet.", err=True)
    raise typer.Exit(code=1)


@app.command()
def download(
    data_dir: DataDirOpt = None,
    track: Track = Track.all,
) -> None:
    """Dual-channel dataset download (Kaggle primary, PyG mirror fallback)."""
    from aml_workbench.data.download import run_download

    try:
        results = run_download(_data_dir(data_dir), dataset=track.value)
    except AmlWorkbenchError as exc:
        typer.echo(f"Fail-closed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    from aml_workbench.data.download import format_result

    for r in results:
        typer.echo(format_result(r))


@app.command()
def ingest(
    data_dir: DataDirOpt = None,
    track: Track = Track.all,
) -> None:
    """Manifest-checksum gate + typed DuckDB ingest + count assertions."""
    from aml_workbench.data.ingest import run_ingest

    root = _data_dir(data_dir)
    try:
        stats = run_ingest(root, track=track.value)
    except AmlWorkbenchError as exc:
        typer.echo(f"Fail-closed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for line in stats:
        typer.echo(line)


@app.command()
def smoke(data_dir: DataDirOpt = None) -> None:
    """LR + RF on the temporal split, gated at ROC-AUC >= 0.80, one-page report."""
    from aml_workbench.smoke import run_smoke

    root = _data_dir(data_dir)
    try:
        report_path = run_smoke(root)
    except AmlWorkbenchError as exc:
        typer.echo(f"Fail-closed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Smoke gate passed; report written to {report_path}")


@app.command()
def rules(data_dir: DataDirOpt = None) -> None:
    """Rules-based scenario engine over HI-Small tables."""
    from aml_workbench.rules import run_rules

    root = _data_dir(data_dir)
    try:
        summary = run_rules(root)
    except AmlWorkbenchError as exc:
        typer.echo(f"Fail-closed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(summary)


@app.command()
def alert_stats(data_dir: DataDirOpt = None) -> None:
    """Per-scenario alert statistics."""
    from aml_workbench.alerts import run_alert_stats

    root = _data_dir(data_dir)
    try:
        report_path = run_alert_stats(root)
    except AmlWorkbenchError as exc:
        typer.echo(f"Fail-closed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Alert stats written to {report_path}")


@app.command()
def graph_features(data_dir: DataDirOpt = None) -> None:
    """NetworkX graph features over the Elliptic temporal graph."""
    from aml_workbench.graph import run_graph_features

    root = _data_dir(data_dir)
    try:
        summary = run_graph_features(root)
    except AmlWorkbenchError as exc:
        typer.echo(f"Fail-closed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(summary)


@app.command()
def baselines(data_dir: DataDirOpt = None) -> None:
    """LR/RF raw-feature baselines on the temporal split."""
    _not_implemented("baselines")


@app.command()
def challenger(data_dir: DataDirOpt = None) -> None:
    """LightGBM challenger vs baselines on predeclared PR-AUC."""
    _not_implemented("challenger")


@app.command()
def shap(data_dir: DataDirOpt = None) -> None:
    """SHAP summary on the best model."""
    _not_implemented("shap")


@app.command()
def validate(data_dir: DataDirOpt = None) -> None:
    """Strict-inductive walk-forward validation + per-timestep metrics."""
    _not_implemented("validate")


@app.command()
def drift(data_dir: DataDirOpt = None) -> None:
    """PSI drift monitoring on scores and features."""
    _not_implemented("drift")


@app.command()
def gnn(data_dir: DataDirOpt = None) -> None:
    """Strict-inductive GraphSAGE baseline (PyG, 3 seeds)."""
    _not_implemented("gnn")


@app.command()
def triage(data_dir: DataDirOpt = None) -> None:
    """Fused rule+ML alert queue + operational KPIs."""
    _not_implemented("triage")


@app.command()
def view(data_dir: DataDirOpt = None) -> None:
    """Thin Streamlit triage view."""
    _not_implemented("view")


@app.command()
def track(data_dir: DataDirOpt = None) -> None:
    """MLflow tracking of model runs."""
    _not_implemented("track")


@app.command()
def report(data_dir: DataDirOpt = None) -> None:
    """Technical report + interview brief + README."""
    _not_implemented("report")


def main() -> None:  # pragma: no cover - thin entry point
    try:
        app()
    except typer.Exit:
        raise
    except AmlWorkbenchError as exc:  # defensive: command bodies already map these
        typer.echo(f"Fail-closed: {exc}", err=True)
        sys.exit(1)
