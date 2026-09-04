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
# mechanically at every ingest.

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

# --- PyG mirror (Elliptic fallback channel) -----------------------------------
# Byte sizes of the served zips, verified 2026-09-02; re-asserted on download.

PYG_MIRROR_ZIP_BYTES: dict[str, int] = {
    "elliptic_txs_features.csv.zip": 150_601_883,
    "elliptic_txs_edgelist.csv.zip": 1_690_631,
    "elliptic_txs_classes.csv.zip": 925_698,
}

# --- Smoke gate -------------------------------------------------------------

SMOKE_ROC_AUC_GATE = 0.80  # dossier: Weber RF ~0.87-0.90; working pipeline clears 0.80
SMOKE_RUNTIME_LIMIT_S = 600.0  # < 10 minutes, fail-closed
SMOKE_SEED = 42

# File-verified 2026-09-03: the released features CSV has 167 fields
# (txId, time_step, 165 feature columns); PyG slices [:, 2:] to a 165-dim x.
# The literature's "166 features" does not match the shipped file - the file wins.

GRAPH_SEED = 42  # Louvain community partition seed (deterministic re-runs)
FEATURE_COUNT = 165
MODEL_SEEDS: tuple[int, ...] = (42, 7, 2026)  # >= 3 seeds recorded for every model
LGBM_N_ESTIMATORS = 400
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 31
# promote the challenger only above this PR-AUC gain over the best baseline
CHALLENGER_MIN_PR_AUC_GAIN = 0.01
SHAP_SAMPLE_ROWS = 20_000  # seeded subsample of test rows for the summary
# --- MLflow tracking ---------------------------------------------------------
MLFLOW_DB_NAME = "mlflow.db"  # sqlite tracking store under the data root

# --- Challenger tuning --------------------------------------------------------
# Deterministic grid, selected on a validation slice carved from TRAIN steps only:
# train 1-30 / validate 31-34 / test 35-49. The test side never enters selection.
TUNING_TRAIN_STEP_MAX = 30
TUNING_VAL_STEP_MIN = 31
TUNING_VAL_STEP_MAX = 34
TUNING_EARLY_STOPPING_ROUNDS = 50
LGBM_GRID: dict[str, tuple[float | int, ...]] = {
    "num_leaves": (15, 31, 63),
    "learning_rate": (0.05, 0.1),
    "min_child_samples": (20, 50),
    "feature_fraction": (0.8, 1.0),
}

# --- Rules scenarios (Track B, HI-Small) -------------------------------
# Scenario thresholds are frozen here and tuned per scenario via
# `aml alert-stats`. Amount comparisons are US Dollar
# only: HI-Small amounts are native-currency and only comparable within one.

REPORTING_THRESHOLD_USD = 10_000.0  # CTR-style reporting threshold (US BSA)
STRUCTURING_MIN_USD = 9_000.0  # "just below" band floor
STRUCTURING_TX_COUNT = 3  # >= 3 sub-threshold payments from one account per day
VELOCITY_TX_COUNT = 20  # >= 20 outgoing transactions in one day (precision-tuned)
# Rapid-churn semantics redefined 2026-09-03: was "payout within 24h of any
# inflow" (CHURN_MAX_DELAY_H, deleted) and now same-day inflow/outflow
# accounting that each sum exactly once — see _churn_sql in rules.py. The
# alert population is NOT comparable to runs before this date.
CHURN_MAX_RETAINED_PCT = 0.10  # round trip keeps <= 10% of the inflow
FAN_MIN_COUNTERPARTIES = 5  # distinct counterparties in one day window
FAN_MIN_AMOUNT_USD = 50_000.0  # aggregate in the same day window
CYCLE_MAX_LENGTH = 3  # bounded cycle search: 2- and 3-cycles only
CYCLE_AMOUNT_TOLERANCE = 0.10  # +/- 10% amount preservation around the cycle
CYCLE_MIN_LEG_USD = 1_000.0  # minimum daily leg amount entering cycle search
COMMUNITY_MIN_SHARED_COUNTERPARTIES = 5  # shared counterparties in one day
COMMUNITY_MAX_HUB_DEGREE = 64  # daily counterparty degree cap before pairing
