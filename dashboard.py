"""
CrediX - Credit Risk, Fraud & Portfolio Analytics Dashboard
Run: streamlit run dashboard.py
"""

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs

st.set_page_config(
    page_title="CrediX | Credit Decisioning & Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2rem; font-weight: 700; color:#0F172A; margin-bottom:.25rem; }
    .sub-header { color:#64748B; margin-bottom:1rem; }
    .section-note { color:#64748B; font-size:.9rem; }
    .badge-clean { background:#ECFDF5; color:#065F46; border:1px solid #A7F3D0; padding:6px 14px; border-radius:999px; font-weight:600; }
    .badge-warn { background:#FFFBEB; color:#92400E; border:1px solid #FDE68A; padding:6px 14px; border-radius:999px; font-weight:600; }
    .badge-critical { background:#FEF2F2; color:#991B1B; border:1px solid #FECACA; padding:6px 14px; border-radius:999px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Portfolio analytics helpers - intentionally kept inside dashboard.py so the
# dashboard works as a drop-in file with the current repository.
# -----------------------------------------------------------------------------

WORKBOOK_NAME = "bank_data_sample_Ammar Elgazar.xlsx"


def _empty_bank_data() -> Dict[str, pd.DataFrame]:
    return {name: pd.DataFrame() for name in ["Customers", "Accounts", "Transactions", "Loans", "Branches"]}


@st.cache_data(show_spinner=False)
def load_bank_data(path: str) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(path):
        return _empty_bank_data()

    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception:
        return _empty_bank_data()

    data = _empty_bank_data()
    for name in data:
        if name in sheets:
            data[name] = sheets[name].copy()
    return data


def prepare_bank_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    date_columns = {
        "Customers": ["date_of_birth", "created_date"],
        "Accounts": ["open_date"],
        "Transactions": ["transaction_date"],
        "Loans": ["start_date"],
        "Branches": [],
    }

    prepared = {}
    for sheet, df in data.items():
        out = df.copy()
        out.columns = [str(c).strip().lower() for c in out.columns]
        for col in date_columns.get(sheet, []):
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], errors="coerce")
        prepared[sheet] = out
    return prepared


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _money(value: float) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}EGP {value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{sign}EGP {value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}EGP {value / 1_000:.1f}K"
    return f"{sign}EGP {value:,.0f}"


def portfolio_kpis(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    customers = data["Customers"]
    accounts = data["Accounts"]
    loans = data["Loans"]
    tx = data["Transactions"]

    active_loans = loans[loans["status"].astype(str).str.lower().eq("active")] if "status" in loans.columns else loans
    active_accounts = accounts[accounts["status"].astype(str).str.lower().eq("active")] if "status" in accounts.columns else accounts

    exposure = _num(active_loans, "principal_amount").sum()
    balances = _num(active_accounts, "balance").sum()
    loan_count = len(active_loans)
    customer_count = customers["customer_id"].nunique() if "customer_id" in customers.columns else len(customers)
    active_account_count = len(active_accounts)
    avg_ticket = exposure / loan_count if loan_count else 0
    avg_rate = _num(active_loans, "interest_rate").mean() if loan_count else 0
    tx_volume = _num(tx, "amount").sum()
    tx_count = len(tx)

    return {
        "customer_count": customer_count,
        "active_account_count": active_account_count,
        "active_loan_count": loan_count,
        "loan_exposure": exposure,
        "deposit_balance": balances,
        "avg_ticket": avg_ticket,
        "avg_rate": avg_rate,
        "transaction_volume": tx_volume,
        "transaction_count": tx_count,
    }


def customer_analytics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["Customers"].copy()
    accounts = data["Accounts"].copy()
    loans = data["Loans"].copy()

    if customers.empty:
        return pd.DataFrame()

    customer_id = "customer_id"
    account_agg = pd.DataFrame(index=customers[customer_id].unique())
    if not accounts.empty and customer_id in accounts.columns:
        a = accounts.copy()
        a["balance"] = _num(a, "balance")
        account_agg = a.groupby(customer_id).agg(
            account_count=("account_id", "nunique") if "account_id" in a.columns else (customer_id, "size"),
            total_balance=("balance", "sum"),
        )

    loan_agg = pd.DataFrame(index=customers[customer_id].unique())
    if not loans.empty and customer_id in loans.columns:
        l = loans.copy()
        l["principal_amount"] = _num(l, "principal_amount")
        loan_agg = l.groupby(customer_id).agg(
            loan_count=("loan_id", "nunique") if "loan_id" in l.columns else (customer_id, "size"),
            total_exposure=("principal_amount", "sum"),
        )

    result = customers.set_index(customer_id).copy()
    result = result.join(account_agg, how="left").join(loan_agg, how="left")
    for col in ["account_count", "loan_count", "total_balance", "total_exposure"]:
        if col not in result:
            result[col] = 0
        result[col] = result[col].fillna(0)

    result["relationship_depth"] = result["account_count"] + result["loan_count"]
    result["customer_segment"] = np.where(result["loan_count"] > 0, "Existing Lending Relationship", "Deposit/Account Only")
    return result.reset_index()


def loan_analytics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    loans = data["Loans"].copy()
    if loans.empty:
        return pd.DataFrame()

    loans["principal_amount"] = _num(loans, "principal_amount")
    loans["interest_rate"] = _num(loans, "interest_rate")
    loans["loan_type"] = loans.get("loan_type", "Unknown").astype(str).str.title()
    loans["status"] = loans.get("status", "Unknown").astype(str).str.title()

    if "tenure_months" in loans.columns:
        loans["tenure_months"] = _num(loans, "tenure_months")

    return loans


def transaction_analytics(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    tx = data["Transactions"].copy()
    accounts = data["Accounts"].copy()

    if tx.empty:
        return pd.DataFrame()

    tx["amount"] = _num(tx, "amount")
    tx["transaction_type"] = tx.get("transaction_type", "Unknown").astype(str).str.lower()
    tx["channel"] = tx.get("channel", "Unknown").astype(str).str.title()

    if not accounts.empty and "account_id" in tx.columns and "account_id" in accounts.columns:
        cols = [c for c in ["account_id", "customer_id"] if c in accounts.columns]
        tx = tx.merge(accounts[cols].drop_duplicates("account_id"), on="account_id", how="left")

    tx["signed_amount"] = np.where(tx["transaction_type"].isin(["withdrawal", "debit", "payment"]), -tx["amount"], tx["amount"])
    if "transaction_date" in tx.columns:
        tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
        tx["month"] = tx["transaction_date"].dt.to_period("M").astype(str)
    return tx


def concentration_table(loans: pd.DataFrame, column: str) -> pd.DataFrame:
    if loans.empty or column not in loans.columns:
        return pd.DataFrame()
    x = loans.groupby(column, dropna=False)["principal_amount"].agg(["count", "sum"]).reset_index()
    x.columns = [column, "Loan Count", "Exposure"]
    total = x["Exposure"].sum()
    x["Exposure Share"] = x["Exposure"] / total if total else 0
    return x.sort_values("Exposure", ascending=False)


def stress_test(exposure: float, base_pd: float, inflation: float, rate_bps: int, unemployment: float, lgd: float = 0.45) -> Dict[str, float]:
    # Transparent scenario assumptions for demo/management analytics only.
    stressed_pd = base_pd * (
        1.0
        + inflation * 0.045
        + (rate_bps / 100.0) * 0.032
        + unemployment * 0.065
    )
    stressed_pd = min(max(stressed_pd, 0.0), 1.0)
    base_el = exposure * base_pd * lgd
    stressed_el = exposure * stressed_pd * lgd
    return {
        "stressed_pd": stressed_pd,
        "base_el": base_el,
        "stressed_el": stressed_el,
        "incremental_el": max(0.0, stressed_el - base_el),
    }


# -----------------------------------------------------------------------------
# Core application setup
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_fraud_engine():
    return CreditFraudEngine()


engine = get_fraud_engine()

st.sidebar.markdown("### 🏦 CrediX Decision Portal")
st.sidebar.caption("Underwriting, Forensic Audit & Portfolio Risk")

preset_options = {
    "Case 1: Returning Customer (Good Standing)": "sample_returning_customer_payload.json",
    "Case 2: New-to-Bank Customer (Cold Start)": "sample_new_to_bank_payload.json",
    "Case 3: Critical Fraud & Document Tampering": "sample_fraudulent_applicant_payload.json",
    "Custom Upload: Choose JSON File": "custom",
}

selected_option = st.sidebar.selectbox("Select Active Application:", list(preset_options.keys()))
payload = None
filename = preset_options[selected_option]

if filename == "custom":
    uploaded_file = st.sidebar.file_uploader("Upload Application JSON (v2 Contract):", type=["json"])
    if uploaded_file is not None:
        try:
            payload = json.load(uploaded_file)
        except Exception as exc:
            st.sidebar.error(f"Error parsing JSON: {exc}")
else:
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        st.sidebar.warning(f"File {filename} not found in workspace.")

if payload is None:
    st.info("Please select or upload an application from the sidebar to inspect.")
    st.stop()

with st.spinner("Processing forensic and predictive features..."):
    assessment = engine.evaluate(payload)
    app_features, history_features = adapt_application_to_model_inputs(payload)

# Load portfolio data independently from the selected application.
CORE_BANKING_FILE = os.path.join(BASE_DIR, WORKBOOK_NAME)
bank_data = prepare_bank_data(load_bank_data(CORE_BANKING_FILE))

# -----------------------------------------------------------------------------
# Application header
# -----------------------------------------------------------------------------

app_id = payload.get("application_id", "N/A")
applicant_name = payload.get("national_id_fields", {}).get("full_name", {}).get("value", "Unspecified Applicant")
loan_purpose = str(payload.get("form_data", {}).get("loan_purpose", "personal_cash")).replace("_", " ").title()
requested_amount = float(payload.get("form_data", {}).get("requested_amount", 0.0) or 0.0)

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"<div class='main-header'>Application: {app_id}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='sub-header'>Applicant: <b>{applicant_name}</b> &nbsp;|&nbsp; Purpose: <b>{loan_purpose}</b> &nbsp;|&nbsp; Facility: <b>EGP {requested_amount:,.0f}</b></div>",
        unsafe_allow_html=True,
    )

with col_badge:
    risk_level = assessment.get("fraud_risk_level", "UNKNOWN")
    if risk_level == "LOW":
        st.markdown("<div class='badge-clean'>✅ Low Fraud Risk (Verified)</div>", unsafe_allow_html=True)
    elif risk_level in ["MEDIUM", "HIGH"]:
        st.markdown(f"<div class='badge-warn'>⚠️ Inconsistency Flagged ({risk_level})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-critical'>🚨 Critical Fraud / Alteration</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Application KPI cards
# -----------------------------------------------------------------------------

metrics = assessment.get("metrics", {})
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    score = float(assessment.get("fraud_risk_score", 0.0) or 0.0)
    st.metric("Ensemble Fraud Score", f"{score:.2f} / 1.00", "Clean Audit" if score <= .25 else "High Alert")
with kpi2:
    mismatch = float(metrics.get("income_mismatch_ratio", 0.0) or 0.0) * 100
    st.metric("Salary vs Inflow Mismatch", f"{mismatch:.1f}%", "Fully Reconciled" if mismatch <= 10 else f"{mismatch:.0f}% Discrepancy")
with kpi3:
    feeder = assessment.get("downstream_risk_feeder", {})
    haircut = float(feeder.get("haircut_percentage", 0.0) or 0.0)
    adj_sal = float(feeder.get("risk_adjusted_salary", 0.0) or 0.0)
    st.metric("Risk-Adjusted Salary", f"EGP {adj_sal:,.0f}", f"-{haircut:.0f}% Haircut" if haircut else "100% Credible")
with kpi4:
    violations = int(metrics.get("total_violations_count", 0) or 0)
    critical = int(metrics.get("critical_violations_count", 0) or 0)
    st.metric("Triggered Audit Rules", f"{violations} Violation(s)", f"{critical} Critical" if critical else "Passed All Rules")

action_code = assessment.get("recommended_action", "REVIEW")
if risk_level == "LOW":
    st.success(f"**Underwriting Action:** `{action_code}` — Fraud screening completed without a high-severity flag.")
elif risk_level in ["MEDIUM", "HIGH"]:
    st.warning(f"**Underwriting Action:** `{action_code}` — Elevated fraud indicators require manual review.")
else:
    st.error(f"**Underwriting Action:** `{action_code}` — Critical fraud indicators detected.")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Summary (XAI)",
    "🤖 Machine Learning Models",
    "🔍 Forensic Audit & CBE Codes",
    "📊 Credit Risk & Financials",
    "📈 Portfolio & Advanced Analytics",
    "💻 Raw Enriched JSON",
])

# TAB 1
with tab1:
    st.subheader("Underwriter AI Narrative (Explainable Intelligence)")
    st.info(assessment.get("explainable_ai", {}).get("executive_summary_en", "No explanation available."))

    st.markdown("### Cross-Document Verification Matrix")
    c1, c2, c3 = st.columns(3)
    checklist = assessment.get("verification_checklist", {})
    with c1:
        st.write("🪪 **National ID Consistency:**", "✅ Verified" if checklist.get("identity_verified") else "❌ Mismatch / Invalid")
        st.write("💵 **Salary Inflow Reconciled:**", "✅ Reconciled" if checklist.get("income_verified") else "❌ Discrepancy Flagged")
    with c2:
        st.write("🏢 **Employer Entity Match:**", "✅ Matched" if checklist.get("employer_verified") else "❌ Entity Discrepancy")
        st.write("📑 **Document Digital Integrity:**", "✅ Authentic" if checklist.get("document_integrity_verified") else "❌ Tampering Suspected")
    with c3:
        st.write("🏛️ **I-Score Bureau Inquiry:**", "✅ Fresh & Active" if checklist.get("bureau_verified") else "❌ Stale / Legal Action")
        st.write("🧮 **Statement Running Balance Math:**", "✅ Balanced" if checklist.get("bank_statement_math_verified") else "❌ Arithmetic Anomaly")

# TAB 2
with tab2:
    st.subheader("Dual Machine Learning Screening Stack")
    st.caption("Supervised fraud probability + unsupervised anomaly screening")
    ml_meta = assessment.get("ml_models_assessment", {})

    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown("#### 🌲 Supervised Model: XGBoost Fraud Classifier")
        xgb_prob = float(ml_meta.get("xgboost_fraud_probability", score) or score)
        st.metric("XGBoost Fraud Probability P(Fraud)", f"{xgb_prob * 100:.1f}%", "Low Risk" if xgb_prob <= .30 else "High Fraud Probability")
        st.progress(min(max(xgb_prob, 0.0), 1.0))
        st.caption("Model output exposed by the fraud engine; use the model card/training pipeline for production calibration details.")
    with ml_col2:
        st.markdown("#### 🔍 Unsupervised Model: Isolation Forest")
        iso_score = float(ml_meta.get("isolation_forest_anomaly_score", score) or score)
        iso_flag = bool(ml_meta.get("isolation_forest_anomaly_detected", risk_level == "CRITICAL"))
        st.metric("Isolation Forest Outlier Score", f"{iso_score:.2f} / 1.00", "Normal Inlier" if not iso_flag else "Multivariate Outlier")
        st.progress(min(max(iso_score, 0.0), 1.0))
        st.caption("Screens multidimensional financial and behavioral anomalies exposed by the fraud engine.")

# TAB 3
with tab3:
    st.subheader("Regulatory Audit Violations & Policy Codes")
    triggered = assessment.get("triggered_rules", [])
    if triggered:
        for rule in triggered:
            with st.expander(f"[{rule.get('severity', 'N/A')}] {rule.get('rule_code', 'N/A')} — {rule.get('rule_name_en', 'Rule')}", expanded=True):
                st.write(f"**Finding:** {rule.get('description_en', '')}")
                a, b = st.columns(2)
                a.caption(f"Observed Value: `{rule.get('observed_value', 'N/A')}`")
                b.caption(f"Policy Threshold: `{rule.get('threshold_value', 'N/A')}`")
    else:
        st.success("✅ Zero policy violations detected across the uploaded documents.")

    st.markdown("---")
    st.subheader("Behavioral Banking Anomalies")
    anomalies = assessment.get("behavioral_anomalies", [])
    if anomalies:
        for anom in anomalies:
            left, right = st.columns([1, 4])
            with left:
                st.write(f"{'🚨' if anom.get('detected') else '🟢'} **{anom.get('anomaly_name', 'Anomaly')}**")
            with right:
                st.write(anom.get("explanation_en", ""))
    else:
        st.info("No behavioral anomaly records returned by the engine.")

# TAB 4
with tab4:
    st.subheader("Financial Standing & Bureau Profile")
    col_cr1, col_cr2 = st.columns(2)
    with col_cr1:
        st.markdown("#### Employment & Income Profile")
        salary_fields = payload.get("salary_certificate_fields", {})
        st.write(f"- **Declared Net Monthly Salary:** EGP {float(salary_fields.get('declared_net_salary', {}).get('value', 0) or 0):,.0f}")
        st.write(f"- **Risk-Adjusted Salary:** EGP {float(feeder.get('risk_adjusted_salary', 0) or 0):,.0f}")
        st.write(f"- **Applicant Age:** {payload.get('national_id_fields', {}).get('age_years', {}).get('value', 'N/A')} Years")
        st.write(f"- **Employer:** {salary_fields.get('employer_name', {}).get('value', 'N/A')}")
        st.write(f"- **Tenure:** {salary_fields.get('employment_tenure_years', {}).get('value', 'N/A')} Years")
    with col_cr2:
        st.markdown("#### Egyptian I-Score & Core Banking Track")
        iscore = payload.get("iscore_report_fields", {})
        st.write(f"- **I-Score Credit Score:** {iscore.get('credit_score', {}).get('value', 'N/A')} ({iscore.get('score_tier', {}).get('value', 'N/A')})")
        st.write(f"- **Bank Relationship:** {'Returning Customer' if payload.get('is_returning_customer') else 'New-to-Bank'}")
        st.write(f"- **Requested Monthly Installment:** EGP {float(payload.get('form_data', {}).get('requested_annuity', 0) or 0):,.0f}")
        st.write(f"- **Requested Tenure:** {payload.get('form_data', {}).get('tenure_months', 0)} Months")

    st.markdown("#### Canonical Features Ready for Downstream Risk Model")
    st.dataframe(pd.DataFrame([app_features]), use_container_width=True)

# TAB 5 - REAL DATA-DRIVEN PORTFOLIO ANALYTICS
with tab5:
    st.subheader("📈 Portfolio Intelligence & Scenario Analytics")
    st.caption("Portfolio metrics are calculated from the repository's core-banking workbook; no placeholder portfolio totals are used.")

    customers = bank_data["Customers"]
    accounts = bank_data["Accounts"]
    loans = loan_analytics(bank_data)
    tx = transaction_analytics(bank_data)
    branches = bank_data["Branches"]
    kpis = portfolio_kpis(bank_data)

    if all(df.empty for df in bank_data.values()):
        st.warning(f"Core-banking workbook not found: `{WORKBOOK_NAME}`")
    else:
        # 1. Portfolio overview
        st.markdown("### 1. Portfolio Overview")
        a, b, c, d, e = st.columns(5)
        a.metric("Customers", f"{int(kpis['customer_count']):,}")
        b.metric("Active Accounts", f"{int(kpis['active_account_count']):,}")
        c.metric("Active Loans", f"{int(kpis['active_loan_count']):,}")
        d.metric("Loan Exposure", _money(kpis["loan_exposure"]))
        e.metric("Account Balances", _money(kpis["deposit_balance"]))

        a, b, c, d = st.columns(4)
        a.metric("Average Loan Ticket", _money(kpis["avg_ticket"]))
        b.metric("Average Interest Rate", f"{kpis['avg_rate'] * 100:.2f}%")
        c.metric("Transactions", f"{int(kpis['transaction_count']):,}")
        d.metric("Transaction Volume", _money(kpis["transaction_volume"]))

        st.markdown("---")

        # 2. Customer relationship analytics
        st.markdown("### 2. Customer Relationship & Exposure")
        cust = customer_analytics(bank_data)
        if cust.empty:
            st.info("Customer-level analytics are unavailable because the Customers sheet is empty.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                segment_counts = cust["customer_segment"].value_counts().rename_axis("Segment").to_frame("Customers")
                st.bar_chart(segment_counts)
            with c2:
                display_cols = [c for c in ["customer_id", "full_name", "kyc_status", "account_count", "total_balance", "loan_count", "total_exposure"] if c in cust.columns]
                st.dataframe(
                    cust[display_cols].sort_values("total_exposure", ascending=False).head(20),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")

        # 3. Credit exposure and concentration
        st.markdown("### 3. Credit Exposure & Concentration")
        if loans.empty:
            st.info("Loan analytics are unavailable because the Loans sheet is empty.")
        else:
            l1, l2 = st.columns(2)
            with l1:
                by_type = concentration_table(loans, "loan_type")
                if not by_type.empty:
                    chart = by_type.set_index("loan_type")["Exposure"]
                    st.bar_chart(chart)
                    st.dataframe(by_type, use_container_width=True, hide_index=True)
            with l2:
                by_status = concentration_table(loans, "status")
                if not by_status.empty:
                    st.bar_chart(by_status.set_index("status")["Exposure"])
                    st.dataframe(by_status, use_container_width=True, hide_index=True)

            if not branches.empty and "branch_id" in loans.columns and "branch_id" in branches.columns:
                pass

            if "customer_id" in loans.columns and not customers.empty and "customer_id" in customers.columns and "branch_id" in customers.columns:
                branch_view = loans.merge(
                    customers[["customer_id", "branch_id"]].drop_duplicates("customer_id"),
                    on="customer_id",
                    how="left",
                )
                if not branches.empty and "branch_id" in branches.columns:
                    branch_view = branch_view.merge(branches, on="branch_id", how="left")
                if "region" in branch_view.columns:
                    by_region = branch_view.groupby("region", dropna=False)["principal_amount"].sum().sort_values(ascending=False)
                    st.markdown("#### Regional Exposure")
                    st.bar_chart(by_region)

        st.markdown("---")

        # 4. Transaction and cashflow behavior
        st.markdown("### 4. Transaction & Cashflow Behavior")
        if tx.empty:
            st.info("Transaction analytics are unavailable because the Transactions sheet is empty.")
        else:
            t1, t2 = st.columns(2)
            with t1:
                type_summary = tx.groupby("transaction_type")["amount"].agg(["count", "sum"]).reset_index()
                type_summary.columns = ["Transaction Type", "Count", "Volume"]
                st.dataframe(type_summary, use_container_width=True, hide_index=True)
                st.bar_chart(type_summary.set_index("Transaction Type")["Volume"])
            with t2:
                channel_summary = tx.groupby("channel")["amount"].agg(["count", "sum"]).reset_index()
                channel_summary.columns = ["Channel", "Count", "Volume"]
                st.dataframe(channel_summary, use_container_width=True, hide_index=True)
                st.bar_chart(channel_summary.set_index("Channel")["Volume"])

            if "month" in tx.columns:
                monthly = tx.dropna(subset=["transaction_date"]).groupby("month")["signed_amount"].sum()
                if not monthly.empty:
                    st.markdown("#### Monthly Net Cash Movement")
                    st.line_chart(monthly)

        st.markdown("---")

        # 5. Scenario stress testing
        st.markdown("### 5. Scenario-Based Portfolio Stress Testing")
        st.caption("Scenario analysis only. Elasticity assumptions are transparent demo parameters and should be calibrated against historical portfolio data before production use.")

        exposure = float(kpis["loan_exposure"])
        base_pd = 0.05
        if "status" in loans.columns and not loans.empty:
            # If the workbook contains a delinquent/default status, use its observed share.
            status_lower = loans["status"].astype(str).str.lower()
            observed_bad = status_lower.isin(["default", "delinquent", "npl", "past_due"]).mean()
            if observed_bad > 0:
                base_pd = float(observed_bad)

        s1, s2 = st.columns([1, 2])
        with s1:
            inflation = st.slider("Inflation Shock (+ percentage points)", 0.0, 12.0, 3.0, 0.5)
            rate_bps = st.slider("Policy Rate Shock (+ bps)", 0, 600, 200, 50)
            unemployment = st.slider("Unemployment Shock (+ percentage points)", 0.0, 8.0, 1.5, 0.5)
            lgd = st.slider("LGD Assumption", 0.10, 0.80, 0.45, 0.05)

        result = stress_test(exposure, base_pd, inflation, rate_bps, unemployment, lgd)
        with s2:
            x1, x2, x3 = st.columns(3)
            x1.metric("Baseline PD", f"{base_pd * 100:.2f}%")
            x2.metric("Stressed PD", f"{result['stressed_pd'] * 100:.2f}%")
            x3.metric("Incremental EL", _money(result["incremental_el"]))

            st.metric("Baseline Expected Loss", _money(result["base_el"]))
            st.metric("Stressed Expected Loss", _money(result["stressed_el"]))

            curve = []
            for shock in np.arange(0, 10.1, 1.0):
                r = stress_test(exposure, base_pd, shock, 0, 0, lgd)
                curve.append({"Inflation Shock": f"+{shock:.0f}%", "Stressed PD": r["stressed_pd"] * 100})
            curve_df = pd.DataFrame(curve).set_index("Inflation Shock")
            st.line_chart(curve_df)

        st.markdown("---")

        # 6. Data quality / availability
        st.markdown("### 6. Analytics Data Quality & Availability")
        quality_rows = []
        for sheet_name, df in bank_data.items():
            quality_rows.append({
                "Sheet": sheet_name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing Cells": int(df.isna().sum().sum()) if not df.empty else 0,
                "Status": "Available" if not df.empty else "Missing / Empty",
            })
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
        st.caption("The dashboard shows data-availability information instead of inventing portfolio metrics when source data are unavailable.")

# TAB 6
with tab6:
    st.subheader("Enriched Contract JSON Payload")
    st.json(engine.enrich_payload(payload))
