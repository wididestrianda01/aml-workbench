"""Frozen configuration: every expected value the pipeline asserts lives here.

The numbers in this module are the numbers the portfolio later cites — they are
checked mechanically at gate time, not remembered. Tests inject violations via
monkeypatching these module attributes at runtime; they never edit this file.

Pinning notes (observed at first real download, 2026-09-03):
- HI-Small observed 5,078,345 transactions (the documented Kaggle figure). The
  planning-stage expectation of >= 5,181,000 was superseded by the observed
  value, documented in data/manifest.json; the gate remains fail-closed below
  the pinned floor.
- HI-Small laundering rate observed 5,177 / 5,078,345 = 1-in-980.9, consistent
  with the paper's 1-in-981 (Altman et al. 2023, Table 3). This resolves the
  Kaggle-description (5.1K) vs paper (3.6K) discrepancy in favor of ~5.2K.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_ENV = "AML_WORKBENCH_DATA_DIR"


def default_data_dir() -> Path:
    """Data root; overridable via AML_WORKBENCH_DATA_DIR (tests use tmp paths)."""
    env = os.environ.get(DATA_DIR_ENV)
    return Path(env) if env else PROJECT_ROOT / "data"


# --- Elliptic (Track A) frozen expectations -----------------------------------
# Verified against primary sources 2026-09-02 (research dossier R2); re-asserted
# mechanically at every ingest (gate C4).

EXPECTED_TX_COUNT = 203_769
EXPECTED_EDGE_COUNT = 234_355
# class map: 1 -> illicit, 2 -> licit, unknown -> NULL
EXPECTED_CLASS_COUNTS: dict[int | None, int] = {1: 4_545, 2: 42_019, None: 157_205}
EXPECTED_TIME_STEPS: frozenset[int] = frozenset(range(1, 50))

# Locked temporal split: train on labeled steps 1-34, test on 35-49. Never random.
TRAIN_STEP_MAX = 34
TEST_STEP_MIN = 35

# --- HI-Small (Track B) gate floors -------------------------------------------
# ">= 5,181,000" was the planning expectation; the actual published file has
# 5,078,345 tx (pinned 2026-09-03, see module docstring + manifest).

HI_SMALL_MIN_TX = 5_078_345
HI_SMALL_MIN_ACCOUNTS = 515_000
HI_SMALL_LAUNDERING_RATE_TARGET = 1 / 981
HI_SMALL_LAUNDERING_RATE_TOLERANCE = 0.05  # +/- 5% band around 1-in-981
HI_SMALL_LAUNDERING_COUNT_PINNED = 5_177  # exact pinned observed count (2026-09-03)

# --- C5 smoke gate -------------------------------------------------------------

SMOKE_ROC_AUC_GATE = 0.80  # dossier: Weber RF ~0.87-0.90; working pipeline clears 0.80
SMOKE_RUNTIME_LIMIT_S = 600.0  # < 10 minutes, fail-closed
SMOKE_SEED = 42

# File-verified 2026-09-03: the released features CSV has 167 fields
# (txId, time_step, 165 feature columns); PyG slices [:, 2:] to a 165-dim x.
# The literature's "166 features" does not match the shipped file - the file wins.
FEATURE_COUNT = 165
