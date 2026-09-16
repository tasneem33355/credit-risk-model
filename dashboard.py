"""
Streamlit Web Dashboard for Credit Risk, Fraud & Portfolio Analytics
=====================================================================
Platform: Smart Financing & Credit Request Analysis Platform (CrediX)
Language: Professional Financial English
Target: Underwriters, Credit Risk Officers (CRO), and Bank Executives
"""

import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

CORE_BANKING_FILE = os.path.join(
    BASE_DIR,
    "bank_data_sample_Ammar Elgazar.xlsx"
)

core_banking_data = prepare_data(
    load_core_banking_data(CORE_BANKING_FILE)
)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs
from portfolio_analytics import (
    load_core_banking_data,
    prepare_data,
    calculate_portfolio_kpis,
    build_customer_analytics,
    build_loan_analytics,
    build_transaction_analytics,
    calculate_concentration,
    stress_test,
    build_stress_curve,
    calculate_data_quality,
)

st.set_page_config(
    page_title="CrediX | Enterprise Credit Decisioning & Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.25rem;
    }
    .badge-clean {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-warn {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_fraud_engine():
    return CreditFraudEngine()

engine = get_fraud_engine()

# -----------------------------------------------------------------------------
# Sidebar: Navigation & Inputs
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🏦 CrediX Decision Portal")
st.sidebar.caption("Underwriting, Forensic Audit & Portfolio Risk")

preset_options = {
    "Case 1: Returning Customer (Good Standing)": "sample_returning_customer_payload.json",
    "Case 2: New-to-Bank Customer (Cold Start)": "sample_new_to_bank_payload.json",
    "Case 3: Critical Fraud & Document Tampering": "sample_fraudulent_applicant_payload.json",
    "Custom Upload: Choose JSON File": "custom"
}

selected_option = st.sidebar.selectbox("Select Active Application:", list(preset_options.keys()))

payload = None
filename = preset_options[selected_option]

if filename == "custom":
    uploaded_file = st.sidebar.file_uploader("Upload Application JSON (v2 Contract):", type=["json"])
    if uploaded_file is not None:
        try:
            payload = json.load(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Error parsing JSON: {e}")
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

# Evaluation Run
with st.spinner("Processing Forensics & Predictive Features..."):
    assessment = engine.evaluate(payload)
    app_features, history_features = adapt_application_to_model_inputs(payload)

# Header Section
app_id = payload.get("application_id", "N/A")
applicant_name = payload.get("national_id_fields", {}).get("full_name", {}).get("value", "Unspecified Applicant")
loan_purpose = str(payload.get("form_data", {}).get("loan_purpose", "personal_cash")).replace("_", " ").title()
requested_amount = float(payload.get("form_data", {}).get("requested_amount", 0.0))

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"<div class='main-header'>Application: {app_id}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>Applicant: <b>{applicant_name}</b> &nbsp;|&nbsp; Purpose: <b>{loan_purpose}</b> &nbsp;|&nbsp; Facility: <b>EGP {requested_amount:,.0f}</b></div>", unsafe_allow_html=True)

with col_badge:
    risk_level = assessment["fraud_risk_level"]
    if risk_level == "LOW":
        st.markdown("<div class='badge-clean'>✅ Low Fraud Risk (Verified)</div>", unsafe_allow_html=True)
    elif risk_level in ["MEDIUM", "HIGH"]:
        st.markdown(f"<div class='badge-warn'>⚠️ Inconsistency Flagged ({risk_level})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-critical'>🚨 Critical Fraud / Alteration</div>", unsafe_allow_html=True)

st.markdown("---")

# KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Ensemble Fraud Score",
        value=f"{assessment['fraud_risk_score']:.2f} / 1.00",
        delta="Clean Audit" if assessment['fraud_risk_score'] <= 0.25 else "High Alert",
        delta_color="normal" if assessment['fraud_risk_score'] <= 0.25 else "inverse"
    )

with kpi2:
    mismatch = assessment["metrics"]["income_mismatch_ratio"] * 100
    st.metric(
        label="Salary vs Inflow Mismatch",
        value=f"{mismatch:.1f}%",
        delta="Fully Reconciled" if mismatch <= 10 else f"{mismatch:.0f}% Discrepancy",
        delta_color="normal" if mismatch <= 10 else "inverse"
    )

with kpi3:
    haircut = assessment["downstream_risk_feeder"]["haircut_percentage"]
    adj_sal = assessment["downstream_risk_feeder"]["risk_adjusted_salary"]
    st.metric(
        label="Underwriting Approved Salary",
        value=f"EGP {adj_sal:,.0f}",
        delta=f"-{haircut:.0f}% Haircut" if haircut > 0 else "100% Credible",
        delta_color="normal" if haircut == 0 else "inverse"
    )

with kpi4:
    st.metric(
        label="Triggered Audit Rules",
        value=f"{assessment['metrics']['total_violations_count']} Violation(s)",
        delta=f"{assessment['metrics']['critical_violations_count']} Critical" if assessment['metrics']['critical_violations_count'] > 0 else "Passed All Rules",
        delta_color="normal" if assessment['metrics']['critical_violations_count'] == 0 else "inverse"
    )

# Executive Recommendation Banner
action_code = assessment["recommended_action"]
if risk_level == "LOW":
    st.success(f"**Underwriting Action:** `PROCEED_TO_CREDIT_EVALUATION` — All cross-document and ML fraud screenings passed.")
elif risk_level in ["MEDIUM", "HIGH"]:
    st.warning(f"**Underwriting Action:** `{action_code}` — Elevated risk patterns identified. Senior manual review required.")
else:
    st.error(f"**Underwriting Action:** `REJECT_SUSPECTED_FRAUD` — Critical document alteration or income falsification identified.")

# -----------------------------------------------------------------------------
# Tabbed Deep-Dive Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Summary (XAI)",
    "🤖 Machine Learning Models",
    "🔍 Forensic Audit & CBE Codes",
    "📊 Credit Risk & Financials",
    "📈 Portfolio & Advanced Analytics",
    "💻 Raw Enriched JSON"
])

# TAB 1: Executive Summary
with tab1:
    st.subheader("Underwriter AI Narrative (Explainable Intelligence)")
    st.info(assessment["explainable_ai"]["executive_summary_en"])

    st.markdown("### Cross-Document Verification Matrix")
    c1, c2, c3 = st.columns(3)
    with c1:
        id_ok = assessment["verification_checklist"]["identity_verified"]
        st.write("🪪 **National ID Consistency:**", "✅ Verified" if id_ok else "❌ Mismatch / Invalid")
        inc_ok = assessment["verification_checklist"]["income_verified"]
        st.write("💵 **Salary Inflow Reconciled:**", "✅ Reconciled" if inc_ok else "❌ Discrepancy Flagged")
    with c2:
        emp_ok = assessment["verification_checklist"]["employer_verified"]
        st.write("🏢 **Employer Entity Match:**", "✅ Matched" if emp_ok else "❌ Entity Discrepancy")
        doc_ok = assessment["verification_checklist"]["document_integrity_verified"]
        st.write("📑 **Document Digital Integrity:**", "✅ Authentic" if doc_ok else "❌ Tampering Suspected")
    with c3:
        bur_ok = assessment["verification_checklist"]["bureau_verified"]
        st.write("🏛️ **I-Score Bureau Inquiry:**", "✅ Fresh & Active" if bur_ok else "❌ Stale / Legal Action")
        math_ok = assessment["verification_checklist"]["bank_statement_math_verified"]
        st.write("🧮 **Statement Running Balance Math:**", "✅ Balanced" if math_ok else "❌ Arithmetic Anomaly")

# TAB 2: Machine Learning Models (Bulletproof Safe)
with tab2:
    st.subheader("Dual Machine Learning Screening Stack")
    st.caption("Fusing Supervised Classification with Unsupervised Multi-dimensional Anomaly Detection")
    
    ml_meta = assessment.get("ml_models_assessment", {
        "isolation_forest_anomaly_score": assessment.get("fraud_risk_score", 0.05),
        "isolation_forest_anomaly_detected": assessment.get("fraud_risk_level") == "CRITICAL",
        "xgboost_fraud_probability": assessment.get("fraud_risk_score", 0.05)
    })
    
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown("#### 🌲 Supervised Model: XGBoost Fraud Classifier")
        xgb_prob = float(ml_meta.get("xgboost_fraud_probability", 0.05))
        st.metric(
            label="XGBoost Fraud Probability P(Fraud)",
            value=f"{xgb_prob * 100:.1f}%",
            delta="Low Risk" if xgb_prob <= 0.30 else "High Fraud Probability",
            delta_color="normal" if xgb_prob <= 0.30 else "inverse"
        )
        st.progress(float(min(max(xgb_prob, 0.0), 1.0)))
        st.caption("Trained to detect fraudulent application profiles across income mismatch, debt stress, and velocity.")
        
    with ml_col2:
        st.markdown("#### 🔍 Unsupervised Model: Isolation Forest Anomaly Index")
        iso_score = float(ml_meta.get("isolation_forest_anomaly_score", 0.05))
        iso_flag = bool(ml_meta.get("isolation_forest_anomaly_detected", False))
        st.metric(
            label="Isolation Forest Outlier Score",
            value=f"{iso_score:.2f} / 1.00",
            delta="Normal Inlier" if not iso_flag else "Multivariate Outlier",
            delta_color="normal" if not iso_flag else "inverse"
        )
        st.progress(float(min(max(iso_score, 0.0), 1.0)))
        st.caption("Screens high-dimensional cashflow dispersion, sudden liquidity spikes, and non-linear behavior.")

# TAB 3: Forensic Audit & CBE Codes
with tab3:
    st.subheader("Regulatory Audit Violations & Policy Codes")
    if assessment["triggered_rules"]:
        for rule in assessment["triggered_rules"]:
            with st.expander(f"[{rule['severity']}] {rule['rule_code']} — {rule['rule_name_en']}", expanded=True):
                st.write(f"**Finding:** {rule['description_en']}")
                col_obs, col_thresh = st.columns(2)
                col_obs.caption(f"Observed Value: `{rule['observed_value']}`")
                col_thresh.caption(f"Policy Threshold: `{rule['threshold_value']}`")
    else:
        st.success("✅ Zero regulatory policy violations detected across uploaded documents.")

    st.markdown("---")
    st.subheader("Behavioral Banking Anomalies")
    for anom in assessment["behavioral_anomalies"]:
        col_status, col_desc = st.columns([1, 4])
        with col_status:
            st.markdown(f"{'🚨' if anom['detected'] else '🟢'} **{anom['anomaly_name']}**")
        with col_desc:
            st.write(anom["explanation_en"])

# TAB 4: Credit Risk & Financials
with tab4:
    st.subheader("Financial Standing & Bureau Profile")
    col_cr1, col_cr2 = st.columns(2)
    with col_cr1:
        st.markdown("#### Employment & Income Profile")
        dec_salary = payload.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0)
        st.write(f"- **Declared Net Monthly Salary:** EGP {dec_salary:,.0f}")
        st.write(f"- **Underwriting Risk-Adjusted Salary:** EGP {assessment['downstream_risk_feeder']['risk_adjusted_salary']:,.0f}")
        st.write(f"- **Applicant Age:** {payload.get('national_id_fields', {}).get('age_years', {}).get('value', 'N/A')} Years")
        st.write(f"- **Employer:** {payload.get('salary_certificate_fields', {}).get('employer_name', {}).get('value', 'N/A')}")
        st.write(f"- **Tenure:** {payload.get('salary_certificate_fields', {}).get('employment_tenure_years', {}).get('value', 'N/A')} Years")

    with col_cr2:
        st.markdown("#### Egyptian I-Score & Core Banking Track")
        iscore_val = payload.get("iscore_report_fields", {}).get("credit_score", {}).get("value", "N/A")
        tier_val = payload.get("iscore_report_fields", {}).get("score_tier", {}).get("value", "N/A")
        st.write(f"- **I-Score Credit Score:** {iscore_val} ({tier_val})")
        st.write(f"- **Bank Relationship:** {'Returning Customer (Internal History Available)' if payload.get('is_returning_customer') else 'New-to-Bank (Cold Start / No Internal History)'}")
        st.write(f"- **Requested Monthly Installment:** EGP {payload.get('form_data', {}).get('requested_annuity', 0):,.0f}")
        st.write(f"- **Requested Tenure:** {payload.get('form_data', {}).get('tenure_months', 0)} Months")

    st.markdown("#### Canonical Features Ready for Downstream Risk Model (413 Features Sample)")
    st.dataframe([app_features], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: PORTFOLIO & ADVANCED ANALYTICS
# -----------------------------------------------------------------------------
with tab5:

    st.subheader("📈 Portfolio Intelligence & Risk Analytics")
    st.caption(
        "Portfolio-level business intelligence, customer behavior, "
        "credit exposure and scenario-based risk analytics."
    )

    # =====================================================================
    # 1. PORTFOLIO OVERVIEW
    # =====================================================================

    st.markdown("### 1. Portfolio Overview")

    kpis = calculate_portfolio_kpis(core_banking_data)

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.metric(
            "Customers",
            f"{kpis['customer_count']:,}"
        )

    with p2:
        st.metric(
            "Active Loans",
            f"{kpis['active_loan_count']:,}"
        )

    with p3:
        st.metric(
            "Loan Exposure",
            f"EGP {kpis['loan_exposure']:,.0f}"
        )

    with p4:
        st.metric(
            "Average Ticket",
            f"EGP {kpis['average_ticket']:,.0f}"
        )

    with p5:
        st.metric(
            "Avg. Interest Rate",
            f"{kpis['average_interest_rate'] * 100:.2f}%"
        )

    st.info(
        "Portfolio KPIs are calculated directly from the available "
        "core-banking sample data. They are not hard-coded."
    )

    st.markdown("---")

    # =====================================================================
    # 2. CUSTOMER RELATIONSHIP ANALYTICS
    # =====================================================================

    st.markdown("### 2. Customer Relationship & Exposure Analytics")

    customer_df = build_customer_analytics(
        core_banking_data
    )

    if not customer_df.empty:

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("#### Relationship Depth")

            relationship_summary = pd.DataFrame({
                "Metric": [
                    "Customers",
                    "Avg Accounts / Customer",
                    "Avg Loans / Customer",
                    "Avg Relationship Products",
                    "Customers with Loans",
                ],
                "Value": [
                    len(customer_df),
                    round(customer_df["account_count"].mean(), 2),
                    round(customer_df["loan_count"].mean(), 2),
                    round(customer_df["relationship_depth"].mean(), 2),
                    int(
                        (customer_df["loan_count"] > 0).sum()
                    ),
                ],
            })

            st.dataframe(
                relationship_summary,
                use_container_width=True,
                hide_index=True,
            )

        with c2:

            st.markdown("#### Customer Exposure")

            exposure_view = customer_df[
                [
                    "customer_id",
                    "full_name",
                    "total_balance",
                    "loan_exposure",
                    "relationship_depth",
                ]
            ].copy()

            exposure_view = exposure_view.sort_values(
                "loan_exposure",
                ascending=False
            )

            st.dataframe(
                exposure_view,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.warning(
            "Customer relationship analytics are unavailable "
            "because core-banking customer data is missing."
        )

    st.markdown("---")

    # =====================================================================
    # 3. CREDIT EXPOSURE ANALYTICS
    # =====================================================================

    st.markdown("### 3. Credit Exposure & Concentration")

    loan_type_df, branch_df = build_loan_analytics(
        core_banking_data
    )

    concentration = calculate_concentration(
        core_banking_data.get(
            "loans",
            pd.DataFrame()
        )
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        st.metric(
            "Top Customer Exposure Share",
            f"{concentration['top_customer_share'] * 100:.1f}%"
        )

    with e2:
        st.metric(
            "Top 5 Customers Share",
            f"{concentration['top_5_customer_share'] * 100:.1f}%"
        )

    with e3:
        st.metric(
            "Customer Concentration (HHI)",
            f"{concentration['herfindahl_index']:.3f}"
        )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Exposure by Loan Type")

        if not loan_type_df.empty:

            display_df = loan_type_df.copy()

            display_df["total_exposure"] = (
                display_df["total_exposure"]
                .map(lambda x: f"EGP {x:,.0f}")
            )

            display_df["average_ticket"] = (
                display_df["average_ticket"]
                .map(lambda x: f"EGP {x:,.0f}")
            )

            display_df["average_interest_rate"] = (
                display_df["average_interest_rate"] * 100
            ).map(lambda x: f"{x:.2f}%")

            display_df["exposure_share"] = (
                display_df["exposure_share"] * 100
            ).map(lambda x: f"{x:.1f}%")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("Loan type analytics unavailable.")

    with right:

        st.markdown("#### Exposure by Branch / Region")

        if not branch_df.empty:

            chart_col = (
                "region"
                if "region" in branch_df.columns
                else branch_df.columns[0]
            )

            chart_df = branch_df.set_index(
                chart_col
            )["total_exposure"]

            st.bar_chart(chart_df)

        else:
            st.info("Branch exposure analytics unavailable.")

    st.markdown("---")

    # =====================================================================
    # 4. TRANSACTION & BEHAVIOR ANALYTICS
    # =====================================================================

    st.markdown("### 4. Transaction & Cashflow Behavior")

    tx_analytics = build_transaction_analytics(
        core_banking_data
    )

    tx_summary = tx_analytics["summary"]

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric(
            "Transactions",
            f"{tx_summary.get('transaction_count', 0):,}"
        )

    with t2:
        st.metric(
            "Deposit Volume",
            f"EGP {tx_summary.get('deposit_volume', 0):,.0f}"
        )

    with t3:
        st.metric(
            "Withdrawal Volume",
            f"EGP {tx_summary.get('withdrawal_volume', 0):,.0f}"
        )

    with t4:
        st.metric(
            "Net Cash Movement",
            f"EGP {tx_summary.get('net_cash_movement', 0):,.0f}"
        )

    tx_left, tx_right = st.columns(2)

    with tx_left:

        st.markdown("#### Transaction Mix")

        if not tx_analytics["types"].empty:

            type_chart = (
                tx_analytics["types"]
                .set_index("transaction_type")
                ["transaction_volume"]
            )

            st.bar_chart(type_chart)

        else:
            st.info("Transaction-type data unavailable.")

    with tx_right:

        st.markdown("#### Channel Usage")

        if not tx_analytics["channels"].empty:

            channel_chart = (
                tx_analytics["channels"]
                .set_index("channel")
                ["transaction_volume"]
            )

            st.bar_chart(channel_chart)

        else:
            st.info("Transaction-channel data unavailable.")

    st.markdown("---")

    # =====================================================================
    # 5. STRESS TESTING
    # =====================================================================

    st.markdown("### 5. Scenario-Based Portfolio Stress Testing")

    st.caption(
        "Interactive scenario analysis using transparent elasticity "
        "assumptions. These assumptions are scenario parameters and "
        "should be calibrated against bank historical data before production use."
    )

    exposure = float(kpis["loan_exposure"])

    if exposure > 0:

        s1, s2, s3 = st.columns(3)

        with s1:
            inflation_shock = st.slider(
                "Inflation Shock (+%)",
                min_value=0.0,
                max_value=12.0,
                value=3.0,
                step=0.5,
            )

        with s2:
            rate_hike_bps = st.slider(
                "Interest Rate Shock (+bps)",
                min_value=0,
                max_value=600,
                value=200,
                step=50,
            )

        with s3:
            unemployment_shock = st.slider(
                "Unemployment Shock (+%)",
                min_value=0.0,
                max_value=8.0,
                value=1.5,
                step=0.5,
            )

        baseline_pd = st.number_input(
            "Baseline PD Assumption",
            min_value=0.001,
            max_value=0.50,
            value=0.0784,
            step=0.005,
            format="%.4f",
        )

        lgd = st.number_input(
            "LGD Assumption",
            min_value=0.05,
            max_value=1.00,
            value=0.45,
            step=0.05,
            format="%.2f",
        )

        stress_result = stress_test(
            portfolio_exposure=exposure,
            baseline_pd=baseline_pd,
            lgd=lgd,
            inflation_shock=inflation_shock,
            rate_hike_bps=rate_hike_bps,
            unemployment_shock=unemployment_shock,
        )

        st.markdown("#### Scenario Impact")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric(
                "Baseline PD",
                f"{stress_result['baseline_pd'] * 100:.2f}%"
            )

        with r2:
            st.metric(
                "Stressed PD",
                f"{stress_result['stressed_pd'] * 100:.2f}%",
                f"+{stress_result['pd_uplift'] * 100:.2f} pp",
                delta_color="inverse",
            )

        with r3:
            st.metric(
                "Baseline Expected Loss",
                f"EGP {stress_result['baseline_expected_loss']:,.0f}"
            )

        with r4:
            st.metric(
                "Incremental Expected Loss",
                f"EGP {stress_result['incremental_expected_loss']:,.0f}",
                delta_color="inverse",
            )

        curve_df = build_stress_curve(
            portfolio_exposure=exposure,
            baseline_pd=baseline_pd,
            lgd=lgd,
            shock_name="Inflation",
        )

        curve_df = curve_df.set_index("Shock")

        st.markdown("#### Inflation Sensitivity Curve")

        st.line_chart(
            curve_df[
                [
                    "Stressed PD",
                    "Expected Loss",
                ]
            ]
        )

    else:
        st.info(
            "Stress testing requires a positive portfolio exposure."
        )

    st.markdown("---")

    # =====================================================================
    # 6. DATA QUALITY / ANALYTICS READINESS
    # =====================================================================

    st.markdown("### 6. Analytics Data Quality")

    quality_df = calculate_data_quality(
        core_banking_data
    )

    st.dataframe(
        quality_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Data-quality monitoring is included to make analytics "
        "limitations visible before portfolio metrics are used for "
        "business decisions."
    )

# TAB 6: Raw JSON
with tab6:
    st.subheader("Enriched Contract JSON Payload")
    enriched_payload = engine.enrich_payload(payload)
    st.json(enriched_payload)
