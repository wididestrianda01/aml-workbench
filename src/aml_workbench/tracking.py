"""Versioned run manifest: code commit, config fingerprint, and the MLflow
run lineage recorded by prior training stages, exposed through `aml track`.

The manifest is the lineage anchor cited by the rollback runbook: every model
number in the technical report must trace to a run id here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.tracking import MlflowClient

from aml_workbench import config
from aml_workbench.errors import DataQualityError

if TYPE_CHECKING:
    from mlflow.entities import Run

MANIFEST_NAME = "run_manifest.json"


def _commit() -> str:
    """Head commit, or 'unknown' when the code is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.PROJECT_ROOT,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=config.PROJECT_ROOT,
    ).stdout.strip()
    return f"{out}+dirty" if dirty else out


def _fingerprint() -> str:
    """Stable SHA-256 over every public config constant (sorted by name).
    repr rather than JSON: config dicts mix None and int keys, which
    json.dumps(sort_keys=True) cannot order."""
    values = {k: v for k, v in vars(config).items() if k.isupper() and not k.startswith("_")}
    blob = repr(sorted(values.items(), key=lambda kv: kv[0])).encode()
    return hashlib.sha256(blob).hexdigest()


def _collect_runs(data_dir: Path) -> list[dict[str, Any]]:
    """Every MLflow run in the local sqlite store, deterministically ordered."""
    store = data_dir / config.MLFLOW_DB_NAME
    if not store.exists():
        raise DataQualityError(
            f"MLflow tracking store not found at {store}; run a training stage first"
        )
    mlflow.set_tracking_uri(f"sqlite:///{store}")
    client = MlflowClient()
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    runs: list[Run] = client.search_runs(experiment_ids, order_by=["attributes.start_time ASC"])
    if not runs:
        raise DataQualityError("MLflow store contains no runs; run a training stage first")
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName", ""),
            "status": r.info.status,
            "start_time_utc": r.info.start_time // 1000,
            "params": dict(sorted(r.data.params.items())),
            "metrics": {k: round(v, 6) for k, v in sorted(r.data.metrics.items())},
        }
        for r in runs
    ]


def run_track(data_dir: Path) -> str:
    """Write the versioned run manifest; fail-closed on missing/empty store."""
    runs = _collect_runs(data_dir)
    manifest = {
        "commit": _commit(),
        "config_fingerprint": _fingerprint(),
        "run_count": len(runs),
        "runs": runs,
    }
    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    commit = str(manifest["commit"])
    return f"track: {len(runs)} runs, commit {commit[:12]} -> {path}"
