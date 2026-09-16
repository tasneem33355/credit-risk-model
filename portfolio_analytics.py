"""
CrediX Portfolio Analytics
==========================

Portfolio-level analytics utilities for the Streamlit dashboard.

The module is intentionally data-source agnostic:
- It can consume the sample core-banking Excel workbook.
- It can also work with empty/missing sheets without crashing.
- It avoids hard-coded portfolio KPIs.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_WORKBOOK = "bank_data_sample_Ammar Elgazar.xlsx"


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def load_core_banking_data(workbook_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load the available core-banking workbook.

    Expected sheets:
        Customers
        Accounts
        Transactions
        Loans
        Branches
    """

    empty = {
        "customers": pd.DataFrame(),
        "accounts": pd.DataFrame(),
        "transactions": pd.DataFrame(),
        "loans": pd.DataFrame(),
        "branches": pd.DataFrame(),
    }

    if not workbook_path or not os.path.exists(workbook_path):
        return empty

    try:
        excel = pd.ExcelFile(workbook_path)
    except Exception:
        return empty

    def read_sheet(name: str) -> pd.DataFrame:
        if name not in excel.sheet_names:
            return pd.DataFrame()

        try:
            return pd.read_excel(workbook_path, sheet_name=name)
        except Exception:
            return pd.DataFrame()

    return {
        "customers": read_sheet("Customers"),
        "accounts": read_sheet("Accounts"),
        "transactions": read_sheet("Transactions"),
        "loans": read_sheet("Loans"),
        "branches": read_sheet("Branches"),
    }


# ---------------------------------------------------------------------
# Cleaning / Preparation
# ---------------------------------------------------------------------

def prepare_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Standardize dates and numeric columns while preserving the source
    structure.
    """

    result = {}

    for name, df in data.items():
        if df is None:
            result[name] = pd.DataFrame()
            continue

        df = df.copy()

        # Normalize column names
        df.columns = [
            str(c).strip().lower().replace(" ", "_")
            for c in df.columns
        ]

        # Parse likely date columns
        for col in df.columns:
            if (
                "date" in col
                or col.endswith("_at")
                or col.endswith("_time")
            ):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        result[name] = df

    return result


# ---------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------

def calculate_portfolio_kpis(
    data: Dict[str, pd.DataFrame]
) -> Dict[str, float]:
    customers = data.get("customers", pd.DataFrame())
    accounts = data.get("accounts", pd.DataFrame())
    loans = data.get("loans", pd.DataFrame())
    transactions = data.get("transactions", pd.DataFrame())

    active_loans = pd.DataFrame()

    if not loans.empty and "status" in loans.columns:
        active_loans = loans[
            loans["status"].astype(str).str.lower().eq("active")
        ].copy()
    else:
        active_loans = loans.copy()

    loan_exposure = 0.0
    average_ticket = 0.0
    average_rate = 0.0

    if not active_loans.empty:
        if "principal_amount" in active_loans.columns:
            loan_exposure = pd.to_numeric(
                active_loans["principal_amount"],
                errors="coerce"
            ).fillna(0).sum()

            average_ticket = pd.to_numeric(
                active_loans["principal_amount"],
                errors="coerce"
            ).mean()

        if "interest_rate" in active_loans.columns:
            average_rate = pd.to_numeric(
                active_loans["interest_rate"],
                errors="coerce"
            ).mean()

    account_balance = 0.0

    if not accounts.empty and "balance" in accounts.columns:
        account_balance = pd.to_numeric(
            accounts["balance"],
            errors="coerce"
        ).fillna(0).sum()

    transaction_volume = 0.0

    if not transactions.empty and "amount" in transactions.columns:
        transaction_volume = pd.to_numeric(
            transactions["amount"],
            errors="coerce"
        ).fillna(0).sum()

    return {
        "customer_count": int(len(customers)),
        "account_count": int(len(accounts)),
        "active_loan_count": int(len(active_loans)),
        "loan_exposure": float(loan_exposure),
        "average_ticket": float(average_ticket) if not np.isnan(average_ticket) else 0.0,
        "average_interest_rate": float(average_rate) if not np.isnan(average_rate) else 0.0,
        "total_account_balance": float(account_balance),
        "transaction_volume": float(transaction_volume),
    }


# ---------------------------------------------------------------------
# Customer Relationship Analytics
# ---------------------------------------------------------------------

def build_customer_analytics(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:

    customers = data.get("customers", pd.DataFrame()).copy()
    accounts = data.get("accounts", pd.DataFrame()).copy()
    loans = data.get("loans", pd.DataFrame()).copy()

    if customers.empty or "customer_id" not in customers.columns:
        return pd.DataFrame()

    result = customers[
        [c for c in [
            "customer_id",
            "full_name",
            "branch_id",
            "kyc_status",
            "created_date",
        ] if c in customers.columns]
    ].copy()

    # Account aggregation
    if not accounts.empty and "customer_id" in accounts.columns:
        account_agg = accounts.groupby("customer_id").agg(
            account_count=("account_id", "count")
            if "account_id" in accounts.columns
            else ("customer_id", "count"),
            total_balance=("balance", "sum")
            if "balance" in accounts.columns
            else ("customer_id", "size"),
        ).reset_index()

        result = result.merge(
            account_agg,
            on="customer_id",
            how="left"
        )

    # Loan aggregation
    if not loans.empty and "customer_id" in loans.columns:
        loan_agg_dict = {}

        if "loan_id" in loans.columns:
            loan_agg_dict["loan_count"] = ("loan_id", "count")

        if "principal_amount" in loans.columns:
            loan_agg_dict["loan_exposure"] = (
                "principal_amount",
                "sum"
            )

        if loan_agg_dict:
            loan_agg = loans.groupby("customer_id").agg(
                **loan_agg_dict
            ).reset_index()

            result = result.merge(
                loan_agg,
                on="customer_id",
                how="left"
            )

    numeric_cols = [
        "account_count",
        "total_balance",
        "loan_count",
        "loan_exposure",
    ]

    for col in numeric_cols:
        if col not in result.columns:
            result[col] = 0.0

        result[col] = pd.to_numeric(
            result[col],
            errors="coerce"
        ).fillna(0)

    result["relationship_depth"] = (
        result["account_count"] + result["loan_count"]
    )

    return result


# ---------------------------------------------------------------------
# Loan Analytics
# ---------------------------------------------------------------------

def build_loan_analytics(
    data: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    loans = data.get("loans", pd.DataFrame()).copy()
    branches = data.get("branches", pd.DataFrame()).copy()

    if loans.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "principal_amount" in loans.columns:
        loans["principal_amount"] = pd.to_numeric(
            loans["principal_amount"],
            errors="coerce"
        ).fillna(0)

    if "interest_rate" in loans.columns:
        loans["interest_rate"] = pd.to_numeric(
            loans["interest_rate"],
            errors="coerce"
        ).fillna(0)

    # By loan type
    if "loan_type" in loans.columns:
        by_type = loans.groupby("loan_type").agg(
            loan_count=("loan_id", "count")
            if "loan_id" in loans.columns
            else ("customer_id", "count"),
            total_exposure=("principal_amount", "sum"),
            average_ticket=("principal_amount", "mean"),
            average_interest_rate=("interest_rate", "mean")
            if "interest_rate" in loans.columns
            else ("principal_amount", "mean"),
        ).reset_index()

        by_type["exposure_share"] = (
            by_type["total_exposure"]
            / max(by_type["total_exposure"].sum(), 1)
        )
    else:
        by_type = pd.DataFrame()

    # Branch concentration
    branch_df = loans.copy()

    if (
        not branches.empty
        and "customer_id" in loans.columns
        and "customer_id" in data.get("customers", pd.DataFrame()).columns
    ):
        customer_branch = data["customers"][
            [c for c in ["customer_id", "branch_id"] if c in data["customers"].columns]
        ].drop_duplicates()

        branch_df = branch_df.merge(
            customer_branch,
            on="customer_id",
            how="left"
        )

        if "branch_id" in branches.columns:
            branch_df = branch_df.merge(
                branches[
                    [c for c in [
                        "branch_id",
                        "branch_name",
                        "region"
                    ] if c in branches.columns]
                ],
                on="branch_id",
                how="left"
            )

    if "region" in branch_df.columns:
        group_col = "region"
    elif "branch_id" in branch_df.columns:
        group_col = "branch_id"
    else:
        group_col = None

    if group_col:
        by_branch = branch_df.groupby(
            group_col,
            dropna=False
        ).agg(
            loan_count=("principal_amount", "count"),
            total_exposure=("principal_amount", "sum"),
            average_ticket=("principal_amount", "mean"),
        ).reset_index()

        by_branch["exposure_share"] = (
            by_branch["total_exposure"]
            / max(by_branch["total_exposure"].sum(), 1)
        )
    else:
        by_branch = pd.DataFrame()

    return by_type, by_branch


# ---------------------------------------------------------------------
# Transaction / Cashflow Analytics
# ---------------------------------------------------------------------

def build_transaction_analytics(
    data: Dict[str, pd.DataFrame]
) -> Dict[str, object]:

    tx = data.get("transactions", pd.DataFrame()).copy()

    if tx.empty:
        return {
            "summary": {},
            "channels": pd.DataFrame(),
            "types": pd.DataFrame(),
        }

    if "amount" in tx.columns:
        tx["amount"] = pd.to_numeric(
            tx["amount"],
            errors="coerce"
        ).fillna(0)

    summary = {
        "transaction_count": int(len(tx)),
        "total_volume": float(tx["amount"].sum())
        if "amount" in tx.columns else 0.0,
    }

    if "transaction_type" in tx.columns and "amount" in tx.columns:

        deposits = tx[
            tx["transaction_type"]
            .astype(str)
            .str.lower()
            .eq("deposit")
        ]

        withdrawals = tx[
            tx["transaction_type"]
            .astype(str)
            .str.lower()
            .eq("withdrawal")
        ]

        deposit_volume = float(deposits["amount"].sum())
        withdrawal_volume = float(withdrawals["amount"].sum())

        summary["deposit_volume"] = deposit_volume
        summary["withdrawal_volume"] = withdrawal_volume
        summary["net_cash_movement"] = (
            deposit_volume - withdrawal_volume
        )

    if "channel" in tx.columns and "amount" in tx.columns:
        channels = tx.groupby("channel").agg(
            transaction_count=("amount", "count"),
            transaction_volume=("amount", "sum"),
        ).reset_index()
    else:
        channels = pd.DataFrame()

    if "transaction_type" in tx.columns and "amount" in tx.columns:
        types = tx.groupby("transaction_type").agg(
            transaction_count=("amount", "count"),
            transaction_volume=("amount", "sum"),
        ).reset_index()
    else:
        types = pd.DataFrame()

    return {
        "summary": summary,
        "channels": channels,
        "types": types,
    }


# ---------------------------------------------------------------------
# Concentration Analytics
# ---------------------------------------------------------------------

def calculate_concentration(
    loans: pd.DataFrame
) -> Dict[str, float]:

    if loans.empty or "principal_amount" not in loans.columns:
        return {
            "top_customer_share": 0.0,
            "top_5_customer_share": 0.0,
            "herfindahl_index": 0.0,
        }

    df = loans.copy()

    df["principal_amount"] = pd.to_numeric(
        df["principal_amount"],
        errors="coerce"
    ).fillna(0)

    if "customer_id" not in df.columns:
        return {
            "top_customer_share": 0.0,
            "top_5_customer_share": 0.0,
            "herfindahl_index": 0.0,
        }

    customer_exposure = (
        df.groupby("customer_id")["principal_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    total = customer_exposure.sum()

    if total <= 0:
        return {
            "top_customer_share": 0.0,
            "top_5_customer_share": 0.0,
            "herfindahl_index": 0.0,
        }

    shares = customer_exposure / total

    return {
        "top_customer_share": float(shares.iloc[0]),
        "top_5_customer_share": float(shares.head(5).sum()),
        "herfindahl_index": float((shares ** 2).sum()),
    }


# ---------------------------------------------------------------------
# Stress Testing
# ---------------------------------------------------------------------

def stress_test(
    portfolio_exposure: float,
    baseline_pd: float,
    lgd: float,
    inflation_shock: float,
    rate_hike_bps: float,
    unemployment_shock: float,
) -> Dict[str, float]:

    # Transparent scenario elasticities.
    # These are scenario assumptions, not calibrated CBE parameters.
    pd_multiplier = (
        1.0
        + inflation_shock * 0.045
        + (rate_hike_bps / 100.0) * 0.032
        + unemployment_shock * 0.065
    )

    stressed_pd = min(
        max(baseline_pd * pd_multiplier, 0.0),
        1.0
    )

    baseline_el = portfolio_exposure * baseline_pd * lgd
    stressed_el = portfolio_exposure * stressed_pd * lgd

    return {
        "baseline_pd": baseline_pd,
        "stressed_pd": stressed_pd,
        "pd_uplift": stressed_pd - baseline_pd,
        "baseline_expected_loss": baseline_el,
        "stressed_expected_loss": stressed_el,
        "incremental_expected_loss": max(
            stressed_el - baseline_el,
            0.0
        ),
        "lgd": lgd,
    }


def build_stress_curve(
    portfolio_exposure: float,
    baseline_pd: float,
    lgd: float,
    shock_name: str = "Inflation",
) -> pd.DataFrame:

    shocks = np.arange(0, 13, 1)

    rows = []

    for shock in shocks:

        if shock_name == "Inflation":
            result = stress_test(
                portfolio_exposure,
                baseline_pd,
                lgd,
                inflation_shock=float(shock),
                rate_hike_bps=0,
                unemployment_shock=0,
            )

        elif shock_name == "Unemployment":
            result = stress_test(
                portfolio_exposure,
                baseline_pd,
                lgd,
                inflation_shock=0,
                rate_hike_bps=0,
                unemployment_shock=float(shock),
            )

        else:
            result = stress_test(
                portfolio_exposure,
                baseline_pd,
                lgd,
                inflation_shock=0,
                rate_hike_bps=float(shock * 50),
                unemployment_shock=0,
            )

        rows.append({
            "Shock": shock,
            "Stressed PD": result["stressed_pd"],
            "Expected Loss": result["stressed_expected_loss"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------

def calculate_data_quality(
    data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:

    rows = []

    for name, df in data.items():

        if df.empty:
            rows.append({
                "Dataset": name.title(),
                "Rows": 0,
                "Columns": 0,
                "Missing Cells %": 0.0,
                "Duplicate Rows": 0,
            })
            continue

        total_cells = max(df.shape[0] * df.shape[1], 1)
        missing_pct = (
            df.isna().sum().sum()
            / total_cells
        ) * 100

        rows.append({
            "Dataset": name.title(),
            "Rows": int(df.shape[0]),
            "Columns": int(df.shape[1]),
            "Missing Cells %": round(missing_pct, 2),
            "Duplicate Rows": int(df.duplicated().sum()),
        })

    return pd.DataFrame(rows)
