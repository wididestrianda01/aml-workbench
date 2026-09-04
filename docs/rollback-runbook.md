# Rollback runbook

Purpose: restore a known-good pipeline state after a bad ingest, a bad model
update, or a corrupt tracking store. The workbench is batch and fail-closed, so
most incidents reduce to "delete the bad artifact, re-run the stage". The run
manifest (`data/reports/run_manifest.json`, written by `aml track`) is the
lineage anchor: it records the code commit, config fingerprint, and MLflow run
ids behind every number the report cites.

## 1. Identify the last known-good state

1. Open `data/reports/run_manifest.json`. Note `commit` (the code version) and
   the run ids of the runs you trust.
2. `git log --oneline` to locate that commit. If the manifest says `+dirty`,
   the artifacts were produced by uncommitted code — regenerate the manifest
   from a clean checkout before trusting any comparison.

## 2. Bad ingest (counts or checksums changed, or a stage failed downstream)

The ingest is deterministic and byte-verified, so recovery is a clean re-run:

```bash
git checkout <good-commit>        # if code drifted
rm data/workbench.duckdb          # drop only the database, never data/manifest.json
uv run aml ingest                 # re-verifies checksums + count gates
uv run aml smoke                  # confirm the C5 gate still passes
```

Never edit `data/manifest.json` to make a failing checksum pass. A checksum
mismatch means the local bytes changed; re-download instead
(`uv run aml download`).

## 3. Bad model update (challenger metrics regressed or a gate fired)

1. `uv run aml track` and compare `config_fingerprint` and the challenger run's
   metrics against the trusted manifest (keep a copy of each good manifest; the
   file is overwritten per run).
2. Roll the code back to the good commit, then re-run only the affected chain:

```bash
git checkout <good-commit>
uv run aml challenger && uv run aml shap && uv run aml validate
```

3. If the decision artifact (`data/reports/decision_report.md`) promoted a
   challenger that later looks wrong, re-run `uv run aml challenger` at the good
   commit; the promotion decision is recomputed from the predeclared PR-AUC
   rule, never hand-edited.

## 4. Corrupt MLflow store

`data/mlflow.db` is a derived artifact. Delete it and re-run the training
stages (`baselines`, `challenger`, `gnn`) to rebuild lineage; model artifacts in
`data/reports/` are the source of truth for the report, not MLflow.

## 5. After any rollback

`uv run pytest` (the seam tests fail closed on contract violations), then
`uv run aml track` to write a fresh manifest tied to the restored commit.
