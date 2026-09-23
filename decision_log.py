"""
CrediX Scored-Application Portfolio Log
========================================
Lightweight, embedded log of every credit decision produced by the trained
PD model (app/model.py: score_application()). portfolio_analytics.py reads
from this log to aggregate *real* model outputs across applications over
time, instead of only ever describing one applicant in isolation.

Mirrors the SQLite pattern already used in fraud_engine.py's
SQLiteEntityStore (same embedded, zero-infrastructure approach), so this
needs no new services -- just a second small .db file, git-ignored the
same way fraud_registry.db already is.

Both entry points call log_decision() after a successful score:
  - dashboard.py   (Institutional Quantitative Risk Lab + AI Assistant tabs)
  - api.py         (POST /api/v1/application/evaluate-end-to-end)
so the portfolio view stays consistent no matter how the application was
scored.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_decisions.db")


def _init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scored_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id TEXT UNIQUE,
                timestamp TEXT,
                credit_score INTEGER,
                probability_of_default REAL,
                decision TEXT,
                risk_tier TEXT,
                requested_amount REAL,
                expected_loss REAL,
                expected_profit REAL,
                fraud_risk_level TEXT,
                source TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sd_time ON scored_decisions(timestamp)")
        conn.commit()


def log_decision(
    application_id: str,
    credit_result: Dict[str, Any],
    fraud_risk_level: Optional[str] = None,
    source: str = "dashboard",
    db_path: Optional[str] = None,
) -> None:
    """Best-effort upsert of one scored decision (one row per
    application_id -- rescoring the same application updates it rather than
    duplicating it, since Streamlit reruns this on every interaction).
    Never raises: a logging failure must never break the underwriting flow.
    """
    if not credit_result or not application_id:
        return

    db_path = db_path or DEFAULT_DB_PATH
    try:
        _init_db(db_path)

        pd_str = str(credit_result.get("probability_of_default", "0%")).rstrip("%").strip()
        pd_value = float(pd_str) / 100.0 if pd_str else 0.0

        business_impact = credit_result.get("business_impact", {}) or {}

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO scored_decisions
                    (application_id, timestamp, credit_score, probability_of_default,
                     decision, risk_tier, requested_amount, expected_loss,
                     expected_profit, fraud_risk_level, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(application_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    credit_score=excluded.credit_score,
                    probability_of_default=excluded.probability_of_default,
                    decision=excluded.decision,
                    risk_tier=excluded.risk_tier,
                    requested_amount=excluded.requested_amount,
                    expected_loss=excluded.expected_loss,
                    expected_profit=excluded.expected_profit,
                    fraud_risk_level=excluded.fraud_risk_level,
                    source=excluded.source
                """,
                (
                    application_id,
                    datetime.now(timezone.utc).isoformat(),
                    credit_result.get("credit_score"),
                    pd_value,
                    credit_result.get("decision"),
                    credit_result.get("risk_tier"),
                    business_impact.get("requested_amount"),
                    business_impact.get("expected_loss_if_default"),
                    business_impact.get("expected_annual_profit_if_performing"),
                    fraud_risk_level,
                    source,
                ),
            )
            conn.commit()
    except Exception:
        # Logging is a nice-to-have for portfolio analytics -- never let it
        # break a live scoring request.
        pass


def load_scored_decisions(db_path: Optional[str] = None, limit: int = 5000):
    """Returns a pandas DataFrame of logged decisions (empty if none yet
    or the DB doesn't exist). Imports pandas lazily so this module stays
    importable from lightweight contexts that don't need it."""
    import pandas as pd

    db_path = db_path or DEFAULT_DB_PATH
    if not os.path.exists(db_path):
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM scored_decisions ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
    except Exception:
        return pd.DataFrame()
