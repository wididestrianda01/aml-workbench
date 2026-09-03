"""C4 — mechanical dataset-drift detection: expected counts, asserted, fail = stop.

Expected values are frozen in ``aml_workbench.config`` and read from that
module at gate time so tests can inject violations by monkeypatching (never by
editing constants). Any mismatch raises DataQualityError before downstream
outputs are written.
"""

from __future__ import annotations

from aml_workbench import config
from aml_workbench.errors import DataQualityError


def assert_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise DataQualityError(f"Count gate failed for {label}: expected {expected}, got {actual}.")


def assert_min_count(actual: int, minimum: int, label: str) -> None:
    if actual < expected_floor(minimum):
        raise DataQualityError(
            f"Count gate failed for {label}: expected >= {minimum}, got {actual}."
        )


def expected_floor(minimum: int) -> int:
    """Hook for readability; keeps assert_min_count trivially honest."""
    return minimum


def assert_elliptic_counts(
    tx_count: int,
    edge_count: int,
    class_counts: dict[int | None, int],
    steps: set[int],
) -> None:
    assert_count(tx_count, config.EXPECTED_TX_COUNT, "elliptic transactions")
    assert_count(edge_count, config.EXPECTED_EDGE_COUNT, "elliptic edges")
    for label, expected in config.EXPECTED_CLASS_COUNTS.items():
        actual = class_counts.get(label)
        name = "unknown" if label is None else ("illicit" if label == 1 else "licit")
        assert_count(actual if actual is not None else -1, expected, f"elliptic class {name}")
    if set(steps) != set(config.EXPECTED_TIME_STEPS):
        missing = sorted(set(config.EXPECTED_TIME_STEPS) - set(steps))
        extra = sorted(set(steps) - set(config.EXPECTED_TIME_STEPS))
        raise DataQualityError(
            f"Time-step gate failed: expected exactly steps {sorted(config.EXPECTED_TIME_STEPS)[0]}"
            f"..{sorted(config.EXPECTED_TIME_STEPS)[-1]}; missing={missing}, unexpected={extra}."
        )


def assert_edge_referential_integrity(orphan_endpoints: int) -> None:
    if orphan_endpoints != 0:
        raise DataQualityError(
            "Edge referential-integrity gate failed: "
            f"{orphan_endpoints} edge endpoint(s) reference unknown txIds."
        )


def assert_class_feature_id_sets_equal(only_in_classes: int, only_in_features: int) -> None:
    if only_in_classes != 0 or only_in_features != 0:
        raise DataQualityError(
            "txId set gate failed: classes-vs-features set difference "
            f"(only in classes: {only_in_classes}, only in features: {only_in_features})."
        )


def assert_hi_small_counts(tx_count: int, account_count: int, laundering_count: int) -> None:
    assert_min_count(tx_count, config.HI_SMALL_MIN_TX, "hi-small transactions")
    assert_min_count(account_count, config.HI_SMALL_MIN_ACCOUNTS, "hi-small distinct accounts")
    rate = laundering_count / tx_count if tx_count else 0.0
    target = config.HI_SMALL_LAUNDERING_RATE_TARGET
    tol = config.HI_SMALL_LAUNDERING_RATE_TOLERANCE
    if not (target * (1 - tol) <= rate <= target * (1 + tol)):
        raise DataQualityError(
            f"Laundering-rate gate failed: expected ~{target:.6f} (+/-{tol:.0%}), got {rate:.6f} "
            f"({laundering_count} flagged of {tx_count})."
        )
    pinned = getattr(config, "HI_SMALL_LAUNDERING_COUNT_PINNED", None)
    if pinned is not None and laundering_count != pinned:
        raise DataQualityError(
            f"Laundering count drifted from the pinned value: "
            f"pinned {pinned}, got {laundering_count}."
        )
