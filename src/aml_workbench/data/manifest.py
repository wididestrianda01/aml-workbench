"""C2 — frozen SHA-256 manifest: load, verify, and (bootstrap-only) record.

The manifest (``data/manifest.json``) is committed. Hashes are recorded once at
the first successful download and never regenerated. Ingest verifies every raw
file (SHA-256 + pinned byte size) before touching the database — any mismatch
fails closed with zero outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aml_workbench.errors import DataQualityError, DownloadError

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class FilePin:
    """A frozen expectation for one raw data file."""

    name: str
    size: int
    sha256: str


def manifest_path(data_dir: Path) -> Path:
    return data_dir / MANIFEST_NAME


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = manifest_path(data_dir)
    if not path.exists():
        raise DataQualityError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise DataQualityError(f"Manifest {path} is not a JSON object.")
    return data


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pins_for(manifest: dict[str, Any], dataset: str) -> dict[str, FilePin]:
    """Frozen pins for a dataset's ingest files: {file name: FilePin}."""
    datasets: Any = manifest.get("datasets", {})
    section = datasets.get(dataset)
    if not section:
        raise DataQualityError(f"Manifest has no entry for dataset '{dataset}'.")
    entries: Any = section.get("files", {})
    pins: dict[str, FilePin] = {}
    for name, entry in entries.items():
        sha = entry.get("sha256")
        size = entry.get("bytes")
        if not sha or size is None:
            raise DataQualityError(
                f"Manifest entry for {dataset}/{name} lacks a frozen sha256/bytes pin; "
                "refusing to ingest against unknown hashes (fail-closed)."
            )
        pins[name] = FilePin(name=name, size=int(size), sha256=str(sha))
    if not pins:
        raise DataQualityError(f"Manifest entry for dataset '{dataset}' lists no files.")
    return pins


def verify_raw_files(pins: dict[str, FilePin], raw_dir: Path, dataset: str) -> None:
    """C2 gate: every pinned file must exist with exact bytes and checksum.

    Raises DataQualityError on the first mismatch; caller must not have written
    any output at this point.
    """
    for name, pin in pins.items():
        path = raw_dir / name
        if not path.exists():
            raise DataQualityError(
                f"Raw file missing for {dataset}/{name}: {path}. Run 'aml download' first."
            )
        actual_bytes = path.stat().st_size
        if actual_bytes != pin.size:
            raise DataQualityError(
                f"Byte-size mismatch for {dataset}/{name}: manifest pins {pin.size} B, "
                f"found {actual_bytes} B."
            )
        actual_sha = sha256_of(path)
        if actual_sha != pin.sha256:
            raise DataQualityError(
                f"Checksum mismatch for {dataset}/{name}: manifest pins {pin.sha256}, "
                f"found {actual_sha}. Corrupted or substituted data never enters the workbench."
            )


def record_downloaded_files(
    data_dir: Path,
    dataset: str,
    results: list[tuple[str, int, str, str]],
    *,
    license_note: str,
    source_note: str,
) -> None:
    """Bootstrap-only manifest update after a successful download.

    For each downloaded file: if the manifest already pins it, verify (drift =
    hard failure). If the section/file is absent (first-ever run), record it.
    Never overwrites an existing pin with a different value.
    """
    path = manifest_path(data_dir)
    manifest: dict[str, Any]
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "datasets": {}}
    datasets: dict[str, Any] = manifest.setdefault("datasets", {})
    section: dict[str, Any] = datasets.setdefault(
        dataset, {"license": license_note, "source": source_note, "files": {}}
    )
    files: dict[str, Any] = section.setdefault("files", {})
    for name, size, sha, channel in results:
        entry = files.get(name)
        if entry is None:
            files[name] = {"bytes": size, "sha256": sha, "channel": channel}
        else:
            if entry.get("sha256") != sha or entry.get("bytes") != size:
                raise DownloadError(
                    f"Downloaded {dataset}/{name} does not match the frozen manifest pin "
                    f"(pinned {entry.get('bytes')} B / {entry.get('sha256')}, "
                    f"got {size} B / {sha}). Possible dataset drift or corruption."
                )
            entry["channel"] = channel
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
