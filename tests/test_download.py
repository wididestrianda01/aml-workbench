"""P1-02 seam tests: C1 dual-channel download, fail-closed.

Channels are monkeypatched at the seam boundary (fetch_* functions); no test
touches the network. No credentials ever appear in tests.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from aml_workbench.cli import app
from aml_workbench.data import download as dl
from aml_workbench.errors import DownloadError
from conftest import build_elliptic_fixture, elliptic_files

runner = CliRunner()


def _elliptic_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"elliptic_bitcoin_dataset/{name}", data)
    return buf.getvalue()


def _mirror_zips(files: dict[str, bytes]) -> dict[str, bytes]:
    zips: dict[str, bytes] = {}
    for name, data in files.items():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, data)
        zips[f"{name}.zip"] = buf.getvalue()
    return zips


def _fail_kaggle_zip(slug: str, dest: Path) -> None:
    raise DownloadError("kaggle channel down (test)")


def _fail_url(url: str, dest: Path) -> None:
    raise DownloadError("mirror channel down (test)")


def test_both_elliptic_channels_fail_exits_nonzero_and_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(dl, "fetch_kaggle_dataset_zip", _fail_kaggle_zip)
    monkeypatch.setattr(dl, "fetch_url", _fail_url)
    result = runner.invoke(app, ["download", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert not (tmp_path / "manifest.json").exists()
    raw = tmp_path / "raw" / "elliptic"
    assert not raw.exists() or not any(raw.iterdir())


def test_kaggle_fails_mirror_size_mismatch_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """Mirror zips at wrong byte sizes must never pass the pinned-size check."""
    files = elliptic_files(build_elliptic_fixture())
    zips = _mirror_zips(files)
    monkeypatch.setattr(dl, "fetch_kaggle_dataset_zip", _fail_kaggle_zip)

    def fake_fetch_url(url: str, dest: Path) -> None:
        dest.write_bytes(zips[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(dl, "fetch_url", fake_fetch_url)
    result = runner.invoke(
        app, ["download", "--data-dir", str(tmp_path), "--dataset", "elliptic"]
    )
    assert result.exit_code == 1
    assert not (tmp_path / "manifest.json").exists()


def test_fallback_with_pinned_sizes_succeeds(tmp_path: Path, monkeypatch) -> None:
    """Kaggle fails, PyG mirror succeeds at pinned byte sizes -> fallback used, exit 0."""
    files = elliptic_files(build_elliptic_fixture(seed=1))
    zips = _mirror_zips(files)
    monkeypatch.setattr(dl, "fetch_kaggle_dataset_zip", _fail_kaggle_zip)
    monkeypatch.setattr(
        dl, "PYG_MIRROR_ZIP_BYTES", {name: len(data) for name, data in zips.items()}
    )

    def fake_fetch_url(url: str, dest: Path) -> None:
        dest.write_bytes(zips[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(dl, "fetch_url", fake_fetch_url)
    result = runner.invoke(
        app, ["download", "--data-dir", str(tmp_path), "--dataset", "elliptic"]
    )
    assert result.exit_code == 0
    raw = tmp_path / "raw" / "elliptic"
    for name in files:
        assert (raw / name).read_bytes() == files[name]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    channel = manifest["datasets"]["elliptic"]["files"]["elliptic_txs_classes.csv"]
    assert channel["channel"] == "pyg_mirror"


def test_hi_small_single_channel_failure_fails_closed(tmp_path: Path, monkeypatch) -> None:
    def fail_file(slug: str, file_name: str, dest: Path) -> None:
        raise DownloadError("kaggle down (test)")

    monkeypatch.setattr(dl, "fetch_kaggle_file", fail_file)
    result = runner.invoke(
        app, ["download", "--data-dir", str(tmp_path), "--dataset", "hi-small"]
    )
    assert result.exit_code == 1
    assert not (tmp_path / "manifest.json").exists()


def test_kaggle_primary_success_used_directly(tmp_path: Path, monkeypatch) -> None:
    """Primary channel success means the mirror is never touched."""
    files = elliptic_files(build_elliptic_fixture(seed=2))
    monkeypatch.setattr(
        dl, "fetch_kaggle_dataset_zip", lambda slug, dest: dest.write_bytes(_elliptic_zip(files))
    )

    def fail_url(url: str, dest: Path) -> None:
        raise AssertionError("fallback channel must not be used when kaggle succeeds")

    monkeypatch.setattr(dl, "fetch_url", fail_url)
    result = runner.invoke(
        app, ["download", "--data-dir", str(tmp_path), "--dataset", "elliptic"]
    )
    assert result.exit_code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    features_entry = manifest["datasets"]["elliptic"]["files"]["elliptic_txs_features.csv"]
    assert features_entry["channel"] == "kaggle"
