"""
Streamlit Web Dashboard for Credit Risk & Application Fraud Analysis
=====================================================================
Platform: Smart Financing & Credit Request Analysis Platform (CrediX)
Language: Professional Financial English
Target: Credit Underwriters, Risk Managers, and Compliance Officers
"""

import streamlit as st
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs

st.set_page_config(
    page_title="CrediX | Enterprise Credit Decisioning & Fraud Engine",
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

# Sidebar: Case Selection
st.sidebar.markdown("### 🏦 CrediX Decision Portal")
st.sidebar.caption("Underwriting & Automated Forensic Audit")

preset_options = {
    "Case 1: Returning Customer (Good Standing)": "sample_returning_customer_payload.json",
    "Case 2: New-to-Bank Customer (Cold Start)": "sample_new_to_bank_payload.json",
    "Case 3: Critical Fraud & Document Tampering": "sample_fraudulent_applicant_payload.json",
    "Custom Upload: Choose JSON File": "custom"
}

selected_option = st.sidebar.selectbox("Select Test Application:", list(preset_options.keys()))

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
    st.info("Please select or upload a credit application from the sidebar to start evaluation.")
    st.stop()

# Run Evaluation
with st.spinner("Executing Forensic Audit & Behavioral Screening..."):
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
    st.markdown(f"<div class='sub-header'>Applicant: <b>{applicant_name}</b> &nbsp;|&nbsp; Purpose: <b>{loan_purpose}</b> &nbsp;|&nbsp; Requested Facility: <b>EGP {requested_amount:,.0f}</b></div>", unsafe_allow_html=True)

with col_badge:
    risk_level = assessment["fraud_risk_level"]
    if risk_level == "LOW":
        st.markdown("<div class='badge-clean'>✅ Low Fraud Risk (Verified)</div>", unsafe_allow_html=True)
    elif risk_level in ["MEDIUM", "HIGH"]:
        st.markdown(f"<div class='badge-warn'>⚠️ Suspected Inconsistency ({risk_level})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-critical'>🚨 Critical Fraud / Alteration</div>", unsafe_allow_html=True)

st.markdown("---")

# KPI Summary Cards (Top Layer)
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
    st.success(f"**Recommended Action:** `PROCEED_TO_CREDIT_EVALUATION` — All cross-document and ML fraud screenings passed.")
elif risk_level in ["MEDIUM", "HIGH"]:
    st.warning(f"**Recommended Action:** `{action_code}` — Elevated risk patterns identified. Senior manual review required.")
else:
    st.error(f"**Recommended Action:** `REJECT_SUSPECTED_FRAUD` — Critical document alteration or income falsification identified.")

# Tabbed Deep-Dive
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Executive Summary (XAI)",
    "🤖 Machine Learning Models",
    "🔍 Forensic Audit & CBE Codes",
    "📊 Credit Risk & Financials",
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

# TAB 2: Machine Learning Models (NEW DEDICATED TAB)
with tab2:
    st.subheader("Dual Machine Learning Screening Stack")
    st.caption("Fusing Supervised Classification with Unsupervised Multi-dimensional Anomaly Detection")
    
    ml_col1, ml_col2 = st.columns(2)
    ml_meta = assessment["ml_models_assessment"]
    
    with ml_col1:
        st.markdown("#### 🌲 Supervised Model: XGBoost Fraud Classifier")
        xgb_prob = ml_meta["xgboost_fraud_probability"]
        st.metric(
            label="XGBoost Fraud Probability P(Fraud)",
            value=f"{xgb_prob * 100:.1f}%",
            delta="Normal" if xgb_prob <= 0.30 else "High Fraud Probability",
            delta_color="normal" if xgb_prob <= 0.30 else "inverse"
        )
        st.progress(float(xgb_prob))
        st.write("Trained to detect fraudulent application profiles across income mismatch, debt stress, and velocity.")
        
    with ml_col2:
        st.markdown("#### 🔍 Unsupervised Model: Isolation Forest Anomaly Index")
        iso_score = ml_meta["isolation_forest_anomaly_score"]
        iso_flag = ml_meta["isolation_forest_anomaly_detected"]
        st.metric(
            label="Isolation Forest Outlier Score",
            value=f"{iso_score:.2f} / 1.00",
            delta="Normal Inlier" if not iso_flag else "Multivariate Outlier",
            delta_color="normal" if not iso_flag else "inverse"
        )
        st.progress(float(iso_score))
        st.write("Screens high-dimensional cashflow dispersion, sudden liquidity spikes, and non-linear behavior.")

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

# TAB 5: Raw JSON
with tab5:
    st.subheader("Enriched Contract JSON Payload")
    enriched_payload = engine.enrich_payload(payload)
    st.json(enriched_payload)
