#!/usr/bin/env python3
"""C1 — scripted dataset download (thin wrapper around `aml download`).

Usage: uv run python scripts/download_data.py [--dataset elliptic|hi-small|all]

Licence terms (also in README):
- Elliptic: CC BY-NC-ND 4.0 — attribution, non-commercial, no derivatives.
  Local use only; data files are NEVER committed (data/ is git-ignored).
- IBM HI-Small: CDLA-Sharing-1.0 — derived artifacts shareable with attribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from a fresh clone (repo root on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aml_workbench.data.download import run_download  # noqa: E402
from aml_workbench.errors import AmlWorkbenchError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["elliptic", "hi-small", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir or Path(__file__).resolve().parents[1] / "data"
    try:
        for result in run_download(data_dir, dataset=args.dataset):
            print(
                f"ok {result.dataset}/{result.name} channel={result.channel} "
                f"bytes={result.size} sha256={result.sha256}"
            )
    except AmlWorkbenchError as exc:
        print(f"Fail-closed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
