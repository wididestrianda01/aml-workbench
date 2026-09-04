# Domain Glossary

Ubiquitous language for aml-workbench. Terms here name the seams; use them exactly.

## Temporal split protocol

The locked, non-random split of the Elliptic timeline. `split_temporal` in
`model.py` is the single seam: `protocol="test"` → train 1–34 / test 35–49;
`protocol="tuning"` → train 1–30 / validation 31–34 (selection slice carved
strictly from training steps). Never random; strict inductive; asserted
fail-closed at the seam, not assumed by callers.

## Workbench store

The single DuckDB database (`workbench.duckdb` under the data root) every
stage reads and writes. `db.py` is its only door: gated opens
(`db.open_workbench`), the canonical path (`db.path`), and cross-stage
artifact checks (`db.require`).

## Alert-day contract

The rules engine (Track B) appends `;window=YYYY-MM-DD` to every alert's
`details`; the triage queue extracts it via `rules.ALERT_DAY_SQL`. Both the
emission side (`_window_sql`) and the extraction side live in `rules.py` so
the rules → triage interface has one home.

## Stage

One pipeline command (`aml <stage>`): a `run_*(data_dir)` function over the
workbench store, fail-closed, producing DuckDB tables / parquet / report
artifacts. The pipeline command is the only test seam.
