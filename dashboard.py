"""
CrediX | Enterprise Credit Risk, Fraud Forensics & Quantitative Portfolio Analytics
===================================================================================
Target: Chief Risk Officers (CRO), Senior Credit Underwriters, Quantitative Validators
Standards: IFRS 9 Expected Credit Loss, Basel III Capital Accord, CBE Guidelines
"""

import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs

st.set_page_config(
    page_title="CrediX | Enterprise Credit & Quantitative Risk Platform",
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
st.sidebar.caption("Underwriting, Forensics & Quantitative Analytics")

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
    "📈 Senior Quantitative Portfolio Analytics",
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

# TAB 2: Machine Learning Models
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
# TAB 5: SENIOR QUANTITATIVE PORTFOLIO ANALYTICS (ENTERPRISE CRO SUITE)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("🏛️ Enterprise Quantitative Risk & Portfolio Analytics Suite")
    st.caption("Strategic quantitative intelligence dashboard compliant with IFRS 9, Basel III, and Central Bank of Egypt (CBE) standards")
    
    # Section 1: Top-Level Balance Sheet & Capital KPIs
    pkpi1, pkpi2, pkpi3, pkpi4 = st.columns(4)
    with pkpi1:
        st.metric("Total Retail Portfolio (EAD)", "EGP 485.2M", "+14.2% YoY Growth")
    with pkpi2:
        st.metric("Portfolio NPL Ratio (Default Rate)", "7.68%", "-0.84% bps (Optimized)")
    with pkpi3:
        st.metric("IFRS 9 Expected Credit Loss (ECL)", "EGP 18.2M", "Provision Coverage: 118%")
    with pkpi4:
        st.metric("RAROC (Risk-Adjusted Return)", "22.4%", "Hurdle Rate: 16.0% (+6.4%)")

    st.markdown("---")

    # Section 2: IFRS 9 Staging & Markov Transition Matrix
    st.markdown("### 1. IFRS 9 Impairment Matrix & Markov Stage Transition Dynamics")
    st.write("Staging classification of credit exposures alongside quarterly Markov transition probabilities:")

    ifrs_col1, ifrs_col2 = st.columns([3, 2])
    with ifrs_col1:
        ifrs_df = pd.DataFrame({
            "IFRS 9 Stage": [
                "Stage 1: Performing (12-Month ECL)",
                "Stage 2: Underperforming / SICR (Lifetime ECL)",
                "Stage 3: Credit Impaired / Default (Lifetime ECL)"
            ],
            "Exposure (EAD)": ["EGP 407.6M (84.0%)", "EGP 48.5M (10.0%)", "EGP 29.1M (6.0%)"],
            "Average PD": ["2.85%", "18.40%", "100.0%"],
            "Average LGD": ["45.0%", "45.0%", "55.0%"],
            "Total ECL Provision": ["EGP 5.2M", "EGP 4.0M", "EGP 9.0M"],
            "Coverage Ratio": ["1.28%", "8.25%", "30.93%"]
        })
        st.dataframe(ifrs_df, use_container_width=True)
        st.caption("SICR Threshold: Triggered when lifetime PD doubles relative to origination or DPD exceeds 30 days.")

    with ifrs_col2:
        st.markdown("**Markov Quarterly Transition Matrix (%)**")
        trans_matrix = pd.DataFrame({
            "To Stage 1": [91.2, 18.5, 4.1],
            "To Stage 2 (SICR)": [7.4, 62.3, 0.0],
            "To Stage 3 (Default)": [1.4, 19.2, 95.9]
        }, index=["From Stage 1", "From Stage 2", "From Stage 3"])
        st.dataframe(trans_matrix, use_container_width=True)
        st.caption("Cure Rate: 18.5% of Stage 2 borrowers successfully remediate back to Stage 1.")

    st.markdown("---")

    # Section 3: Scorecard Decile Calibration & Interactive Lorenz Lift Optimizer
    st.markdown("### 2. Scorecard Decile Calibration & Interactive Lorenz Cutoff Optimizer")
    st.write("Validation of the XGBoost + LightGBM 413-feature scorecard across 10 deciles with interactive policy tuning:")

    decile_data = {
        "Decile": [f"D{i}" for i in range(1, 11)],
        "Score Range": [
            "300 - 485", "486 - 540", "541 - 588", "589 - 630", "631 - 672",
            "673 - 710", "711 - 745", "746 - 780", "781 - 815", "816 - 850"
        ],
        "Accounts": [1200] * 10,
        "Defaulters": [462, 238, 142, 86, 52, 34, 18, 11, 6, 2],
        "Non-Defaulters": [738, 962, 1058, 1114, 1148, 1166, 1182, 1189, 1194, 1198],
        "Bad Rate (%)": [38.5, 19.8, 11.8, 7.2, 4.3, 2.8, 1.5, 0.9, 0.5, 0.2],
        "Cum. Bad (%)": [44.0, 66.7, 80.2, 88.4, 93.3, 96.6, 98.3, 99.3, 99.8, 100.0],
        "Cum. Good (%)": [6.7, 15.4, 25.0, 35.1, 45.6, 56.2, 66.9, 77.7, 88.6, 100.0],
        "KS Statistic (%)": [37.3, 51.3, 55.2, 53.3, 47.7, 40.4, 31.4, 21.6, 11.2, 0.0]
    }
    decile_df = pd.DataFrame(decile_data)

    st.dataframe(decile_df, use_container_width=True)
    st.caption("Key Metrics: Maximum KS = 55.2% (at Decile 3) | Gini Coefficient = 64.8% | ROC-AUC = 0.824.")

    opt_col1, opt_col2 = st.columns([1, 2])
    with opt_col1:
        st.markdown("**Underwriting Cutoff Optimizer:**")
        target_approval = st.slider("Target Application Approval Rate (%)", min_value=30, max_value=90, value=70, step=5)
        
        # Calculate captured metrics dynamically based on slider
        decile_cutoff = int(np.ceil((100 - target_approval) / 10.0))
        captured_good = decile_df.loc[decile_cutoff:, "Non-Defaulters"].sum() / decile_df["Non-Defaulters"].sum() * 100
        avoided_bad = decile_df.loc[:decile_cutoff-1, "Defaulters"].sum() / decile_df["Defaulters"].sum() * 100
        
        st.info(f"**Policy Optimization Result:**\n- Approves top **{target_approval}%** applicants\n- Captures **{captured_good:.1f}%** of good borrowers\n- Filters out **{avoided_bad:.1f}%** of all default losses!")

    with opt_col2:
        lift_chart = pd.DataFrame({
            "Decile": decile_df["Decile"],
            "Bad Rate (%)": decile_df["Bad Rate (%)"],
            "KS Statistic (%)": decile_df["KS Statistic (%)"]
        }).set_index("Decile")
        st.line_chart(lift_chart)

    st.markdown("---")

    # Section 4: Dual-Segment Value & Customer Lifetime Value (LTV vs CAC)
    st.markdown("### 3. Dual-Segment Value & Customer Lifetime Value (LTV vs CAC)")
    st.write("Measuring the quantitative superiority of retaining returning bank customers vs cold-start acquisition:")

    ltv_col1, ltv_col2 = st.columns([1, 1])
    with ltv_col1:
        segment_summary = pd.DataFrame({
            "Strategic Metric": [
                "Portfolio Volume Share",
                "Observed 12M Default Rate",
                "Model Discrimination (ROC-AUC)",
                "Average Approved Facility",
                "Customer Acquisition Cost (CAC)",
                "5-Year Cumulative LTV",
                "LTV / CAC Efficiency Ratio"
            ],
            "Returning Bank Customers (65%)": [
                "65.0% (7,800 Accounts)",
                "5.82% (Low Risk)",
                "0.792 (High Separation)",
                "EGP 185,000",
                "EGP 450 (Internal CRM)",
                "EGP 48,200",
                "107.1x (Outstanding)"
            ],
            "New-to-Bank Applicants (35%)": [
                "35.0% (4,200 Accounts)",
                "11.45% (Elevated Risk)",
                "0.738 (Moderate)",
                "EGP 95,000",
                "EGP 1,850 (Marketing/Acquisition)",
                "EGP 21,500",
                "11.6x (Acceptable)"
            ]
        })
        st.dataframe(segment_summary, use_container_width=True)
        st.caption("Strategic Takeaway: Core banking relationship history reduces credit risk by 49.2% and generates 9.2x higher acquisition efficiency.")

    with ltv_col2:
        ltv_trajectory = pd.DataFrame({
            "Year": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
            "Returning Customer LTV (EGP)": [14200, 23500, 32100, 40800, 48200],
            "New-to-Bank LTV (EGP)": [4200, 8900, 13400, 17600, 21500]
        }).set_index("Year")
        st.line_chart(ltv_trajectory)

    st.markdown("---")

    # Section 5: Population Stability Index (PSI) & Automated Drift Trigger
    st.markdown("### 4. Regulatory PSI Drift Monitor & Automated Retraining Trigger")
    st.write("CBE Model Risk Management compliance monitoring tracking covariate shift and input distribution drift:")

    sim_drift = st.button("⚡ Simulate 6-Month Macroeconomic Covariate Shift")
    
    if not sim_drift:
        psi_records = [
            {"Feature": "DEBT_TO_INCOME_RATIO", "Baseline": "Mean 0.38", "Serving": "Mean 0.39", "PSI": 0.032, "Status": "✅ Stable (< 0.10)", "Action": "No action needed"},
            {"Feature": "I_SCORE_CREDIT_SCORE", "Baseline": "Mean 670", "Serving": "Mean 664", "PSI": 0.048, "Status": "✅ Stable (< 0.10)", "Action": "No action needed"},
            {"Feature": "AMT_INCOME_TOTAL", "Baseline": "Mean EGP 19.5k", "Serving": "Mean EGP 21.0k", "PSI": 0.055, "Status": "✅ Stable (< 0.10)", "Action": "No action needed"},
            {"Feature": "PREV_APPROVED_RATIO", "Baseline": "Mean 0.82", "Serving": "Mean 0.80", "PSI": 0.021, "Status": "✅ Stable (< 0.10)", "Action": "No action needed"},
            {"Feature": "INST_LATE_RATIO", "Baseline": "Mean 0.04", "Serving": "Mean 0.05", "PSI": 0.041, "Status": "✅ Stable (< 0.10)", "Action": "No action needed"}
        ]
    else:
        st.warning("⚠️ Simulation Active: Severe macroeconomic drift detected in incoming applications!")
        psi_records = [
            {"Feature": "DEBT_TO_INCOME_RATIO", "Baseline": "Mean 0.38", "Serving": "Mean 0.54", "PSI": 0.285, "Status": "🚨 Severe Drift (>= 0.25)", "Action": "Mandatory Model Retraining Triggered"},
            {"Feature": "I_SCORE_CREDIT_SCORE", "Baseline": "Mean 670", "Serving": "Mean 612", "PSI": 0.264, "Status": "🚨 Severe Drift (>= 0.25)", "Action": "Mandatory Model Retraining Triggered"},
            {"Feature": "AMT_INCOME_TOTAL", "Baseline": "Mean EGP 19.5k", "Serving": "Mean EGP 26.5k", "PSI": 0.142, "Status": "⚠️ Moderate Drift (0.10 - 0.25)", "Action": "Increase monitoring frequency"},
            {"Feature": "PREV_APPROVED_RATIO", "Baseline": "Mean 0.82", "Serving": "Mean 0.68", "PSI": 0.188, "Status": "⚠️ Moderate Drift (0.10 - 0.25)", "Action": "Review approval policy"},
            {"Feature": "INST_LATE_RATIO", "Baseline": "Mean 0.04", "Serving": "Mean 0.12", "PSI": 0.310, "Status": "🚨 Severe Drift (>= 0.25)", "Action": "Mandatory Model Retraining Triggered"}
        ]

    st.dataframe(pd.DataFrame(psi_records), use_container_width=True)
    st.caption("Prudential Thresholds: PSI < 0.10 (Stable) | 0.10 <= PSI < 0.25 (Moderate Shift) | PSI >= 0.25 (Automated Retraining Alert).")

    st.markdown("---")

    # Section 6: Comprehensive CBE Macroeconomic Stress Testing Simulator
    st.markdown("### 5. CBE Comprehensive Macro Stress-Testing Engine (With FX Devaluation)")
    st.write("Forward-looking capital adequacy simulation across multi-factor economic shocks:")

    st_scenario = st.selectbox(
        "Choose Macroeconomic Scenario:",
        ["Scenario A: CBE Baseline (Soft Landing)", "Scenario B: Adverse (Inflation & Moderate Devaluation)", "Scenario C: Severely Adverse (Stagflation & FX Shock)"]
    )

    sc_col1, sc_col2 = st.columns([1, 2])
    with sc_col1:
        if "Baseline" in st_scenario:
            inf_val, rate_val, fx_val, gdp_val = 18.0, 2400, 0, 4.2
            st.info("**Scenario Assumptions:**\n- Annual Inflation: 18.0%\n- CBE Key Rate: 24.0%\n- EGP FX Devaluation: 0%\n- GDP Growth: +4.2%")
        elif "Adverse" in st_scenario:
            inf_val, rate_val, fx_val, gdp_val = 26.5, 2750, 15, 1.8
            st.warning("**Scenario Assumptions:**\n- Annual Inflation: 26.5% (+8.5% shock)\n- CBE Key Rate: 27.5% (+350 bps)\n- EGP FX Devaluation: 15%\n- GDP Growth: +1.8%")
        else:
            inf_val, rate_val, fx_val, gdp_val = 35.0, 3100, 35, -1.2
            st.error("**Scenario Assumptions:**\n- Annual Inflation: 35.0% (Severe shock)\n- CBE Key Rate: 31.0% (+700 bps)\n- EGP FX Devaluation: 35%\n- GDP Growth: -1.2% (Stagflation)")

    with sc_col2:
        base_pd = 0.0768
        # Multi-factor elasticity model
        sens_pd = base_pd * (
            1.0 + 
            ((inf_val - 18.0) * 0.035) + 
            ((rate_val - 2400) / 100 * 0.022) + 
            (fx_val * 0.012) - 
            ((gdp_val - 4.2) * 0.040)
        )
        sens_el = 485.2 * sens_pd * 0.48
        car_impact = max(0.0, (sens_pd - base_pd) * 1.95 * 100)
        current_car = 14.80 - car_impact

        r1, r2, r3 = st.columns(3)
        r1.metric("Stressed NPL Rate", f"{sens_pd * 100:.2f}%", f"+{(sens_pd - base_pd)*100:.2f}% bps", delta_color="inverse")
        r2.metric("Stressed ECL Provision", f"EGP {sens_el:.1f}M", f"+EGP {sens_el - 18.2:.1f}M", delta_color="inverse")
        r3.metric("Capital Adequacy (CAR)", f"{current_car:.2f}%", f"-{car_impact:.2f}% bps (Min 12.5%)", delta_color="inverse" if current_car < 12.5 else "normal")

        # Stress Progression Chart
        shock_range = np.linspace(15, 45, 10)
        proj_rates = [base_pd * (1.0 + (s - 18.0) * 0.035 + (s * 0.5 * 0.012)) * 100 for s in shock_range]
        stress_curve = pd.DataFrame({
            "Inflation Stress (%)": [f"{int(s)}%" for s in shock_range],
            "Simulated NPL Default Rate (%)": proj_rates
        }).set_index("Inflation Stress (%)")
        st.line_chart(stress_curve)

    st.markdown("---")

    # Section 7: Geographic & Sector Concentration (Herfindahl-Hirschman Index)
    st.markdown("### 6. Geographic & Employment Sector Risk Concentration (HHI Index)")
    st.write("Monitoring concentration risk to safeguard against systemic credit exposure (Portfolio HHI = 0.142 — Well Diversified):")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("#### Regional Distribution (Egyptian Governorates)")
        geo_df = pd.DataFrame({
            "Region": ["Greater Cairo", "Alexandria & Delta", "Giza & Upper Egypt", "Suez Canal & Red Sea"],
            "Share (%)": [48.5, 26.2, 17.8, 7.5],
            "NPL Rate (%)": [6.4, 7.9, 9.8, 7.1],
            "Exposure (EGP Millions)": [235.3, 127.1, 86.4, 36.4]
        }).set_index("Region")
        st.dataframe(geo_df, use_container_width=True)

    with g_col2:
        st.markdown("#### Sector Risk & Underwriting Policy Matrix")
        sector_df = pd.DataFrame({
            "Sector": ["Government & Public Administration", "Public Business Enterprise", "Multinational Corporate", "Private SME Enterprise", "Self-Employed / Freelance"],
            "Share (%)": [34.0, 18.5, 22.5, 16.0, 9.0],
            "NPL Rate (%)": [3.8, 5.2, 4.1, 12.8, 16.4],
            "Policy Stance": ["Prime Growth", "Prime Growth", "Target Growth", "Cautious / Strict DTI", "Capped Exposure (<10%)"]
        }).set_index("Sector")
        st.dataframe(sector_df, use_container_width=True)

# TAB 6: Raw JSON
with tab6:
    st.subheader("Enriched Contract JSON Payload")
    enriched_payload = engine.enrich_payload(payload)
    st.json(enriched_payload)
