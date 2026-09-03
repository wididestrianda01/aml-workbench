"""Rules-based scenario engine over the typed HI-Small
tables, exposed through the `aml rules` pipeline command.

Design decisions (documented, not hidden):
- **Tumbling daily windows** (`date_trunc('day', tx_time)`) instead of sliding
  windows: each scenario evaluates per calendar day, which keeps alerts
  deduplicated and deterministic. A campaign straddling midnight splits across
  two day windows — an accepted simplification for a scenario engine.
- **US Dollar only for amount comparisons** (`payment_currency = 'US Dollar'`):
  HI-Small amounts are in the transacted currency and only comparable within
  one. Count-based scenarios (velocity, fan-in/out, community) span currencies.
- **Bounded cycle search**: circular-flow detection covers 2- and 3-cycles with
  a +/- 10% amount-preservation tolerance (config CYCLE_*). Each cycle emits
  ONE alert, canonicalized to its lexicographically smallest participant (the
  same flow seen from each rotation would otherwise alert 2-3 times). Longer
  cycles are out of scope until `alert-stats` shows the bounded search is too
  coarse.
- **Dense-community scale decision**: the counterparty-overlap self-join is
  quadratic in counterparty degree, so counterparties whose daily degree
  exceeds COMMUNITY_MAX_HUB_DEGREE (64) are treated as processing
  infrastructure and excluded from pairing (edges are also pre-deduplicated
  per account-counterparty-day). This bounds the join without changing the
  typology signal: communities are tight clusters, not hub traffic.
- **Alert model** (spec Implementation Decisions): every scenario writes rows
  of (scenario, entity, reason, details) into the `rule_alert` table — the
  schema later fused with ML scores in the Phase-7 triage queue.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from aml_workbench import config
from aml_workbench.errors import DataQualityError

_DAY = "date_trunc('day', tx_time)"

_USD_TXS = """
WITH usd AS (
    SELECT * FROM hismall_transaction
    WHERE payment_currency = 'US Dollar'
)
"""

_ALL_TXS = "WITH all_tx AS (SELECT * FROM hismall_transaction)"



def _structuring_sql() -> str:
    min_usd = config.STRUCTURING_MIN_USD
    threshold = config.REPORTING_THRESHOLD_USD
    min_txs = config.STRUCTURING_TX_COUNT
    return f"""
{_USD_TXS}
SELECT
    'structuring' AS scenario,
    from_account AS entity,
    'multiple sub-threshold payments in one day (split across counterparties)' AS reason,
    'txs=' || count(*)::VARCHAR
    || ';counterparties=' || count(DISTINCT to_account)::VARCHAR
    || ';total_usd=' || round(sum(amount_paid), 2)::VARCHAR
    || ';band_usd=[{min_usd:.0f},{threshold:.0f})'
    || ';window=' || strftime({_DAY}, '%Y-%m-%d') AS details
FROM usd
WHERE amount_paid >= {min_usd} AND amount_paid < {threshold}
GROUP BY from_account, {_DAY}
HAVING count(*) >= {min_txs} AND sum(amount_paid) >= {threshold}
"""


def _velocity_sql() -> str:
    return f"""
{_ALL_TXS}
SELECT
    'velocity' AS scenario,
    from_account AS entity,
    'high outgoing transaction count in one day' AS reason,
    'txs=' || count(*)::VARCHAR
    || ';window=' || strftime({_DAY}, '%Y-%m-%d') AS details
FROM all_tx
GROUP BY from_account, {_DAY}
HAVING count(*) >= {config.VELOCITY_TX_COUNT}
"""


def _churn_sql() -> str:
    max_retained = config.CHURN_MAX_RETAINED_PCT
    max_delay = config.CHURN_MAX_DELAY_H
    return f"""
{_USD_TXS}
, inflow AS (
    SELECT row_number() OVER (ORDER BY tx_time, from_account, to_account) AS in_tx,
           to_account AS acct,
           tx_time,
           amount_received,
           {_DAY} AS day
    FROM usd
),
outflow AS (
    SELECT from_account AS acct, tx_time, amount_paid FROM usd
),
matched AS (
    SELECT
        i.acct,
        i.day,
        i.amount_received,
        sum(o.amount_paid) AS paid_out,
        sum(o.amount_paid) / i.amount_received AS payout_ratio
    FROM inflow i
    JOIN outflow o
      ON o.acct = i.acct
     AND o.tx_time > i.tx_time
     AND o.tx_time <= i.tx_time + INTERVAL {max_delay} HOUR
    GROUP BY i.in_tx, i.acct, i.day, i.amount_received
    HAVING sum(o.amount_paid) >= (1 - {max_retained}) * i.amount_received
)
SELECT
    'rapid-churn' AS scenario,
    acct AS entity,
    'inflow paid out within {max_delay} h with minimal retention' AS reason,
    'received_usd=' || round(amount_received, 2)::VARCHAR
    || ';paid_out_usd=' || round(paid_out, 2)::VARCHAR
    || ';retained_pct=' || round(100 * (1 - payout_ratio), 1)::VARCHAR
    || ';window=' || strftime(day, '%Y-%m-%d') AS details
FROM matched
QUALIFY row_number() OVER (
    PARTITION BY acct, day ORDER BY payout_ratio DESC
) = 1
"""


def _fan_in_sql() -> str:
    return f"""
{_USD_TXS}
SELECT
    'fan-in' AS scenario,
    to_account AS entity,
    'many distinct counterparties paying into one account in one day' AS reason,
    'counterparties=' || count(DISTINCT from_account)::VARCHAR
    || ';total_usd=' || round(sum(amount_paid), 2)::VARCHAR
    || ';window=' || strftime({_DAY}, '%Y-%m-%d') AS details
FROM usd
GROUP BY to_account, {_DAY}
HAVING count(DISTINCT from_account) >= {config.FAN_MIN_COUNTERPARTIES}
   AND sum(amount_paid) >= {config.FAN_MIN_AMOUNT_USD}
"""


def _fan_out_sql() -> str:
    return f"""
{_USD_TXS}
SELECT
    'fan-out' AS scenario,
    from_account AS entity,
    'one account paying out to many distinct counterparties in one day' AS reason,
    'counterparties=' || count(DISTINCT to_account)::VARCHAR
    || ';total_usd=' || round(sum(amount_paid), 2)::VARCHAR
    || ';window=' || strftime({_DAY}, '%Y-%m-%d') AS details
FROM usd
GROUP BY from_account, {_DAY}
HAVING count(DISTINCT to_account) >= {config.FAN_MIN_COUNTERPARTIES}
   AND sum(amount_paid) >= {config.FAN_MIN_AMOUNT_USD}
"""


def _circular_sql() -> str:
    tol = config.CYCLE_AMOUNT_TOLERANCE
    return f"""
{_USD_TXS}
, edge AS (
    SELECT {_DAY} AS day, from_bank, from_account, to_bank, to_account,
           sum(amount_paid) AS amt, count(*) AS txs
    FROM usd
    WHERE amount_paid >= {config.CYCLE_MIN_LEG_USD}
    GROUP BY 1, 2, 3, 4, 5
),
two_cycle AS (
    SELECT
        e1.day,
        least(e1.from_account, e1.to_account) AS entity,
        CASE WHEN e1.from_account <= e1.to_account
             THEN e1.from_account || '->' || e1.to_account || '->' || e1.from_account
             ELSE e1.to_account || '->' || e1.from_account || '->' || e1.to_account
        END AS path,
        CASE WHEN e1.from_account <= e1.to_account
             THEN e1.amt ELSE e2.amt END AS amt_out,
        CASE WHEN e1.from_account <= e1.to_account
             THEN e2.amt ELSE e1.amt END AS amt_back,
        2 AS cycle_len
    FROM edge e1
    JOIN edge e2
      ON e1.day = e2.day
     AND e1.from_bank = e2.to_bank AND e1.from_account = e2.to_account
     AND e1.to_bank = e2.from_bank AND e1.to_account = e2.from_account
    WHERE e2.amt BETWEEN e1.amt * (1 - {tol}) AND e1.amt * (1 + {tol})
),
three_cycle AS (
    SELECT
        e1.day,
        least(e1.from_account, e1.to_account, e2.to_account) AS entity,
        CASE
            WHEN e1.from_account = least(e1.from_account, e1.to_account, e2.to_account)
                THEN e1.from_account || '->' || e1.to_account
                     || '->' || e2.to_account || '->' || e1.from_account
            WHEN e1.to_account = least(e1.from_account, e1.to_account, e2.to_account)
                THEN e1.to_account || '->' || e2.to_account
                     || '->' || e1.from_account || '->' || e1.to_account
            ELSE e2.to_account || '->' || e1.from_account
                 || '->' || e1.to_account || '->' || e2.to_account
        END AS path,
        CASE
            WHEN e1.from_account = least(e1.from_account, e1.to_account, e2.to_account) THEN e1.amt
            WHEN e1.to_account = least(e1.from_account, e1.to_account, e2.to_account) THEN e2.amt
            ELSE e3.amt
        END AS amt_out,
        CASE
            WHEN e1.from_account = least(e1.from_account, e1.to_account, e2.to_account) THEN e3.amt
            WHEN e1.to_account = least(e1.from_account, e1.to_account, e2.to_account) THEN e1.amt
            ELSE e2.amt
        END AS amt_back,
        3 AS cycle_len
    FROM edge e1
    JOIN edge e2
      ON e1.day = e2.day
     AND e1.to_bank = e2.from_bank AND e1.to_account = e2.from_account
    JOIN edge e3
      ON e2.day = e3.day
     AND e2.to_bank = e3.from_bank AND e2.to_account = e3.from_account
     AND e3.to_bank = e1.from_bank AND e3.to_account = e1.from_account
    WHERE e3.amt BETWEEN e1.amt * (1 - {tol}) AND e1.amt * (1 + {tol})
),
cycles AS (
    SELECT * FROM two_cycle
    UNION ALL
    SELECT * FROM three_cycle
)
SELECT DISTINCT
    'circular-flow' AS scenario,
    entity,
    'round-trip flow: funds cycle back within one day' AS reason,
    'path=' || path
    || ';amt_out_usd=' || round(amt_out, 2)::VARCHAR
    || ';amt_back_usd=' || round(amt_back, 2)::VARCHAR
    || ';cycle_len=' || cycle_len::VARCHAR
    || ';window=' || strftime(day, '%Y-%m-%d') AS details
FROM cycles
"""



def _dense_community_sql() -> str:
    return f"""
{_ALL_TXS}
, edge AS (
    SELECT DISTINCT {_DAY} AS day, from_account AS acct, to_account AS cpty FROM all_tx
    UNION
    SELECT DISTINCT {_DAY}, to_account, from_account FROM all_tx
),
deg AS (
    SELECT day, cpty, count(*) AS daily_degree FROM edge GROUP BY 1, 2
),
nonhub AS (
    SELECT e.day, e.acct, e.cpty
    FROM edge e
    JOIN deg g ON e.day = g.day AND e.cpty = g.cpty
    WHERE g.daily_degree <= {config.COMMUNITY_MAX_HUB_DEGREE}
),
pairs AS (
    SELECT e1.acct, e2.acct AS other_acct, e1.day, count(*) AS shared
    FROM nonhub e1
    JOIN nonhub e2
      ON e1.day = e2.day
     AND e1.cpty = e2.cpty
     AND e1.acct < e2.acct
    GROUP BY e1.acct, e2.acct, e1.day
    HAVING count(*) >= {config.COMMUNITY_MIN_SHARED_COUNTERPARTIES}
)
SELECT
    'dense-community' AS scenario,
    acct AS entity,
    'two accounts share an abnormal number of counterparties in one day' AS reason,
    'pair=' || acct || '|' || other_acct
    || ';shared_counterparties=' || shared::VARCHAR
    || ';window=' || strftime(day, '%Y-%m-%d') AS details
FROM pairs
"""


_SCENARIOS: tuple[str, ...] = (
    _structuring_sql(),
    _velocity_sql(),
    _churn_sql(),
    _fan_in_sql(),
    _fan_out_sql(),
    _circular_sql(),
    _dense_community_sql(),
)


def _one(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Fetch a single integer scalar (count queries); fail-closed on absence."""
    rows = con.execute(sql).fetchall()
    return int(rows[0][0])


def _open_workbench(data_dir: Path) -> duckdb.DuckDBPyConnection:
    db_path = data_dir / "workbench.duckdb"
    if not db_path.exists():
        raise DataQualityError(
            f"workbench database not found at {db_path}; run `aml ingest` first"
        )
    con = duckdb.connect(str(db_path))
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "hismall_transaction" not in tables:
        con.close()
        raise DataQualityError(
            "hismall_transaction table missing; run `aml ingest --track hi-small` first"
        )
    if _one(con, "SELECT count(*) FROM hismall_transaction") == 0:
        con.close()
        raise DataQualityError("hismall_transaction is empty; refusing to run scenarios")
    return con


def run_rules(data_dir: Path) -> str:
    """Evaluate every scenario and write the `rule_alert` table. Fail-closed:
    missing database/tables raise DataQualityError before any output exists."""
    con = _open_workbench(data_dir)
    try:
        con.execute(
            "CREATE OR REPLACE TABLE rule_alert AS "
            + "\nUNION ALL\n".join(f"({sql})" for sql in _SCENARIOS)
        )
        columns = {
            row[0] for row in con.execute("DESCRIBE rule_alert").fetchall()
        }
        if columns != {"scenario", "entity", "reason", "details"}:
            raise DataQualityError(
                f"rule_alert schema invariant violated: got {sorted(columns)}"
            )
        alert_count = _one(con, "SELECT count(*) FROM rule_alert")
        scenario_count = _one(
            con, "SELECT count(DISTINCT scenario) FROM rule_alert"
        )
    finally:
        con.close()
    return (
        f"Rules run: {alert_count} alerts across {scenario_count} scenarios; "
        f"alert table rule_alert written"
    )
