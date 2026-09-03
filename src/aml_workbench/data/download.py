"""Dual-channel dataset download, fail-closed.

Elliptic: Kaggle (primary; honors optional KAGGLE_USERNAME/KAGGLE_KEY from the
environment only) with automatic PyG-mirror fallback (auth-free, byte-pinned
zips). HI-Small: Kaggle file-level download (single channel). If every channel
for a dataset fails, the command raises DownloadError — the pipeline stops, no
silent substitution, no partial data passed downstream.

Licence terms: Elliptic CC BY-NC-ND 4.0 (attribution, local use, NO
redistribution — hence data/ is git-ignored). HI-Small CDLA-Sharing-1.0.
"""

from __future__ import annotations

import base64
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from aml_workbench.data.manifest import record_downloaded_files, sha256_of
from aml_workbench.errors import DownloadError

KAGGLE_DOWNLOAD_BASE = "https://www.kaggle.com/api/v1/datasets/download"
PYG_MIRROR_BASE = "https://data.pyg.org/datasets/elliptic"

ELLIPTIC_SLUG = "ellipticco/elliptic-data-set"
HI_SMALL_SLUG = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

ELLIPTIC_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)
HI_SMALL_FILES = ("HI-Small_Trans.csv", "HI-Small_accounts.csv")

# PyG mirror serves zips; byte sizes pinned in the spec/manifest (verified 2026-09-02).
PYG_MIRROR_ZIP_BYTES = {
    "elliptic_txs_features.csv.zip": 150_601_883,
    "elliptic_txs_edgelist.csv.zip": 1_690_631,
    "elliptic_txs_classes.csv.zip": 925_698,
}

ELLIPTIC_LICENSE = (
    "CC BY-NC-ND 4.0 (attribution, non-commercial, no derivatives; local use only - "
    "data/ is git-ignored for this reason)"
)
HI_SMALL_LICENSE = "CDLA-Sharing-1.0 (derived artifacts shareable with attribution)"


@dataclass(frozen=True)
class DownloadResult:
    dataset: str
    name: str
    channel: str
    size: int
    sha256: str


def format_result(result: DownloadResult) -> str:
    """One shared result-line format for the CLI."""
    return (
        f"ok {result.dataset}/{result.name} channel={result.channel} "
        f"bytes={result.size} sha256={result.sha256}"
    )


# --- fetchers (module-level so seam tests can monkeypatch channels) ----------


def fetch_kaggle_dataset_zip(slug: str, dest: Path) -> None:
    """Download a whole Kaggle dataset zip (auth optional; public datasets work)."""
    _fetch(f"{KAGGLE_DOWNLOAD_BASE}/{slug}", dest)


def fetch_kaggle_file(slug: str, file_name: str, dest: Path) -> None:
    """Download one file out of a Kaggle dataset."""
    _fetch(f"{KAGGLE_DOWNLOAD_BASE}/{slug}?fileName={file_name}", dest)


def fetch_url(url: str, dest: Path) -> None:
    """Plain HTTP fetch (PyG mirror channel)."""
    _fetch(url, dest, extra_headers={})


def _fetch(url: str, dest: Path, extra_headers: dict[str, str] | None = None) -> None:
    # Kaggle credentials from the environment ONLY — never code or config.
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    headers: dict[str, str] = dict(extra_headers or {})
    if user and key:
        token = base64.b64encode(f"{user}:{key}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    req = urlrequest.Request(url, headers=headers)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlrequest.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
    except urlerror.HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} fetching {url}") from exc
    except OSError as exc:
        raise DownloadError(f"Network error fetching {url}: {exc}") from exc


# --- per-dataset orchestration ------------------------------------------------


def _extract_zip_member(archive: Path, file_name: str, dest: Path) -> None:
    """Extract the single member whose basename equals file_name (Kaggle nests
    members in a folder; mirror zips may not)."""
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if Path(m).name == file_name]
        if len(members) != 1:
            raise DownloadError(
                f"Archive {archive.name} does not contain exactly one '{file_name}' member."
            )
        dest.write_bytes(zf.read(members[0]))


def _download_elliptic(raw_dir: Path) -> list[DownloadResult]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="aml-elliptic-") as tmp:
        tmp_dir = Path(tmp)
        try:
            archive = tmp_dir / "elliptic_kaggle.zip"
            fetch_kaggle_dataset_zip(ELLIPTIC_SLUG, archive)
            for name in ELLIPTIC_FILES:
                _extract_zip_member(archive, name, tmp_dir / name)
            return _finalize(raw_dir, "elliptic", ELLIPTIC_FILES, tmp_dir, "kaggle")
        except DownloadError as exc:
            failures.append(f"kaggle: {exc}")
        # Channel 2 (fallback): PyG mirror, per-file zips at pinned byte sizes.
        try:
            for name in ELLIPTIC_FILES:
                zip_name = f"{name}.zip"
                pinned = PYG_MIRROR_ZIP_BYTES[zip_name]
                archive = tmp_dir / zip_name
                fetch_url(f"{PYG_MIRROR_BASE}/{zip_name}", archive)
                actual = archive.stat().st_size
                if actual != pinned:
                    raise DownloadError(
                        f"PyG mirror zip {zip_name}: expected {pinned} B, got {actual} B."
                    )
                _extract_zip_member(archive, name, tmp_dir / name)
            return _finalize(raw_dir, "elliptic", ELLIPTIC_FILES, tmp_dir, "pyg_mirror")
        except DownloadError as exc:
            failures.append(f"pyg_mirror: {exc}")
    raise DownloadError(
        "Both Elliptic download channels failed; pipeline stops (no silent substitution). "
        f"Channel failures: {'; '.join(failures)}"
    )


def _download_hi_small(raw_dir: Path) -> list[DownloadResult]:
    with tempfile.TemporaryDirectory(prefix="aml-hismall-") as tmp:
        tmp_dir = Path(tmp)
        try:
            for name in HI_SMALL_FILES:
                fetch_kaggle_file(HI_SMALL_SLUG, name, tmp_dir / name)
            return _finalize(raw_dir, "hi-small", HI_SMALL_FILES, tmp_dir, "kaggle")
        except DownloadError as exc:
            raise DownloadError(
                f"HI-Small download failed (Kaggle is the only channel for this dataset); "
                f"pipeline stops. Failure: {exc}"
            ) from exc


def _finalize(
    raw_dir: Path,
    dataset: str,
    names: tuple[str, ...],
    staged: Path,
    channel: str,
) -> list[DownloadResult]:
    """Move staged files into place only after ALL files succeeded (no partials)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    for name in names:
        src = staged / name
        if not src.exists():
            raise DownloadError(f"Download incomplete: {dataset}/{name} missing from staging.")
        dest = raw_dir / name
        dest.write_bytes(src.read_bytes())
        results.append(
            DownloadResult(
                dataset=dataset,
                name=name,
                channel=channel,
                size=dest.stat().st_size,
                sha256=sha256_of(dest),
            )
        )
    return results


def run_download(
    data_dir: Path,
    dataset: str = "all",
    *,
    skip_manifest_record: bool = False,
) -> list[DownloadResult]:
    """Download the requested datasets and record/verify frozen manifest pins."""
    results: list[DownloadResult] = []
    if dataset in ("elliptic", "all"):
        results += _download_elliptic(data_dir / "raw" / "elliptic")
    if dataset in ("hi-small", "all"):
        results += _download_hi_small(data_dir / "raw" / "hi-small")
    if not skip_manifest_record:
        if dataset in ("elliptic", "all"):
            record_downloaded_files(
                data_dir,
                "elliptic",
                [
                    (r.name, r.size, r.sha256, r.channel)
                    for r in results
                    if r.dataset == "elliptic"
                ],
                license_note=ELLIPTIC_LICENSE,
                source_note=(
                    f"Kaggle {ELLIPTIC_SLUG} (primary); "
                    f"PyG mirror {PYG_MIRROR_BASE} (fallback)"
                ),
            )
        if dataset in ("hi-small", "all"):
            record_downloaded_files(
                data_dir,
                "hi-small",
                [(r.name, r.size, r.sha256, r.channel) for r in results if r.dataset == "hi-small"],
                license_note=HI_SMALL_LICENSE,
                source_note=f"Kaggle {HI_SMALL_SLUG}",
            )
    return results
