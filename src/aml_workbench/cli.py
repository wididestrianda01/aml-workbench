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
    help="AML Workbench — batch pipeline commands (fail-closed).",
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


def _not_implemented(stage: str, phase: int) -> None:
    typer.echo(
        f"Fail-closed: stage '{stage}' is not implemented yet "
        f"(planned for Phase {phase} of the build plan).",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def download(
    data_dir: DataDirOpt = None,
    track: Track = Track.all,
) -> None:
    """C1: dual-channel dataset download (Kaggle primary, PyG mirror fallback)."""
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
    """C2-C4: manifest-checksum gate + typed DuckDB ingest + count assertions."""
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
    """C5: LR + RF on the temporal split, gated at ROC-AUC >= 0.80, one-page report."""
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
    """Phase 2: rules-based scenario engine over HI-Small tables."""
    _not_implemented("rules", 2)


@app.command()
def alert_stats(data_dir: DataDirOpt = None) -> None:
    """Phase 2: per-scenario alert statistics."""
    _not_implemented("alert-stats", 2)


@app.command()
def graph_features(data_dir: DataDirOpt = None) -> None:
    """Phase 3: NetworkX graph features over the Elliptic temporal graph."""
    _not_implemented("graph-features", 3)


@app.command()
def baselines(data_dir: DataDirOpt = None) -> None:
    """Phase 4: LR/RF raw-feature baselines on the temporal split."""
    _not_implemented("baselines", 4)


@app.command()
def challenger(data_dir: DataDirOpt = None) -> None:
    """Phase 4: LightGBM challenger vs baselines on predeclared PR-AUC."""
    _not_implemented("challenger", 4)


@app.command()
def shap(data_dir: DataDirOpt = None) -> None:
    """Phase 4: SHAP summary on the best model."""
    _not_implemented("shap", 4)


@app.command()
def validate(data_dir: DataDirOpt = None) -> None:
    """Phase 5: strict-inductive walk-forward validation + per-timestep metrics."""
    _not_implemented("validate", 5)


@app.command()
def drift(data_dir: DataDirOpt = None) -> None:
    """Phase 5: PSI drift monitoring on scores and features."""
    _not_implemented("drift", 5)


@app.command()
def gnn(data_dir: DataDirOpt = None) -> None:
    """Phase 6: strict-inductive GraphSAGE baseline (PyG, 3 seeds)."""
    _not_implemented("gnn", 6)


@app.command()
def triage(data_dir: DataDirOpt = None) -> None:
    """Phase 7: fused rule+ML alert queue + operational KPIs."""
    _not_implemented("triage", 7)


@app.command()
def view(data_dir: DataDirOpt = None) -> None:
    """Phase 7: thin Streamlit triage view."""
    _not_implemented("view", 7)


@app.command()
def track(data_dir: DataDirOpt = None) -> None:
    """Phase 8: MLflow tracking of model runs."""
    _not_implemented("track", 8)


@app.command()
def report(data_dir: DataDirOpt = None) -> None:
    """Phase 8: technical report + interview brief + README."""
    _not_implemented("report", 8)


def main() -> None:  # pragma: no cover - thin entry point
    try:
        app()
    except typer.Exit:
        raise
    except AmlWorkbenchError as exc:  # defensive: command bodies already map these
        typer.echo(f"Fail-closed: {exc}", err=True)
        sys.exit(1)
