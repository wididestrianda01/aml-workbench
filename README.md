# AML Workbench — AML Transaction Monitoring & Illicit-Activity Detection Workbench

A production-shaped, locally deployable AML transaction-monitoring workbench on
public data — **not** a live banking system. Two tracks:

- **Track A — Elliptic**: real labeled Bitcoin transaction graph (203,769 tx /
  234,355 edges / 165 features / 49 time steps; file-verified — the
  literature's "166 features" does not match the shipped CSV) carrying graph features, ML
  baselines, strict-inductive temporal walk-forward validation, and a GNN baseline.
- **Track B — IBM IT-AML HI-Small**: synthetic bank-like data (~5.1M tx / ~518K
  accounts) carrying the rules-based scenario engine, SQL typology analytics,
  alert triage, and operational KPIs.

## Quick start

```bash
uv sync                # install (uv + PEP 621)
uv run aml --help      # list pipeline stages
uv run aml download    # C1: dual-channel dataset download
uv run aml ingest      # C2-C4: checksum gate + typed DuckDB ingest + count gates
uv run aml smoke       # C5: smoke run + one-page report
uv run pytest          # seam tests
```

Optional Kaggle credentials (both datasets are public; credentials are read from
the environment only, never from code or config): copy `.env.example` to `.env`
and set `KAGGLE_USERNAME` / `KAGGLE_KEY` from a free account at
<https://www.kaggle.com/account>.

## Data & licences (terms respected, stated honestly)

| Dataset | Source | Licence | Consequence |
|---|---|---|---|
| Elliptic (Track A) | Kaggle `ellipticco/elliptic-data-set` (primary), PyG mirror `data.pyg.org/datasets/elliptic/` (fallback) | **CC BY-NC-ND 4.0** | Attribution required; **no redistribution** — therefore `data/` is git-ignored and only the checksum manifest (`data/manifest.json`) is committed. Local use only. |
| IBM HI-Small (Track B) | Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | **CDLA-Sharing-1.0** | Derived artifacts shareable with attribution. |

If every download channel for a dataset fails, the pipeline exits non-zero and
stops — no silent substitution of data, ever.

## Limitation statement

Elliptic is real curated Bitcoin data frozen since 2019 (IP-anonymized,
non-recomputable features); HI-Small is AMLSim-synthetic bank-like data. Neither
is representative of a bank's customer-transaction environment. AML Workbench demonstrates
monitoring methods, validation discipline, and controls on public data.
