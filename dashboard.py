"""
Streamlit Web Dashboard for Credit Risk & Application Fraud Analysis
=====================================================================
Platform: Smart Financing & Credit Request Analysis Platform (CrediX / ZAWOLF)
Language: Professional Financial English
Target: Credit Underwriters, Risk Managers, and Compliance Officers
"""

import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np

# Ensure local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs

# Page configuration
st.set_page_config(
    page_title="CrediX | Enterprise Credit Decisioning & Fraud Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern enterprise financial UI
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
    
    .callout-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0 20px 0;
        font-size: 0.92rem;
        color: #1E293B;
    }
    
    .disclaimer-badge {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Fraud Engine
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

# Run Evaluation using the exact production signatures
with st.spinner("Executing Forensic Audit & Behavioral Screening..."):
    assessment = engine.evaluate(payload)
    app_features, history_features = adapt_application_to_model_inputs(payload)
    combined_features = {**app_features, **history_features}

# Safe metadata retrieval matching v2 JSON contract
app_id = payload.get("application_id", "N/A")
applicant_name = payload.get("national_id_fields", {}).get("full_name", {}).get("value", "Unspecified Applicant")
loan_purpose = str(payload.get("form_data", {}).get("loan_purpose", "personal_cash")).replace("_", " ").title()
requested_amount = float(payload.get("form_data", {}).get("requested_amount", 0.0))

# Header Section
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown(f"<div class='main-header'>Application: {app_id}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>Applicant: <b>{applicant_name}</b> &nbsp;|&nbsp; Purpose: <b>{loan_purpose}</b> &nbsp;|&nbsp; Requested Facility: <b>EGP {requested_amount:,.0f}</b></div>", unsafe_allow_html=True)

with col_badge:
    risk_level = assessment["fraud_risk_level"]
    fraud_score = assessment["fraud_risk_score"]
    if risk_level == "LOW":
        st.markdown(f"<div class='badge-clean'>✅ Low Fraud Risk ({fraud_score:.2f})</div>", unsafe_allow_html=True)
    elif risk_level in ["MEDIUM", "HIGH"]:
        st.markdown(f"<div class='badge-warn'>⚠️ Suspected Inconsistency ({risk_level})</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='badge-critical'>🚨 Critical Fraud / Alteration</div>", unsafe_allow_html=True)

st.markdown("---")

# KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Fraud Risk Score",
        value=f"{assessment['fraud_risk_score']:.2f} / 1.00",
        delta="Clean Audit" if assessment['fraud_risk_score'] <= 0.25 else "High Alert",
        delta_color="normal" if assessment['fraud_risk_score'] <= 0.25 else "inverse"
    )

with kpi2:
    # Convert decimal ratio to percentage (e.g., 0.25 -> 25.0%)
    mismatch_ratio = float(assessment["metrics"].get("income_mismatch_ratio", 0.0))
    mismatch = mismatch_ratio * 100.0
    st.metric(
        label="Salary vs Bank Inflow Mismatch",
        value=f"{mismatch:.1f}%",
        delta="Fully Reconciled" if mismatch <= 10.0 else f"{mismatch:.0f}% Discrepancy",
        delta_color="normal" if mismatch <= 10.0 else "inverse"
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
    st.success(f"**Recommended Underwriting Action:** `PROCEED_TO_CREDIT_EVALUATION` — Application verified with clean forensic trail.")
elif risk_level in ["MEDIUM", "HIGH"]:
    st.warning(f"**Recommended Underwriting Action:** `{action_code}` — Discrepancies detected. Manual officer review required.")
else:
    st.error(f"**Recommended Underwriting Action:** `REJECT_SUSPECTED_FRAUD` — Fatal document alteration or income inflation identified.")

# Tabbed Deep-Dive
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Executive Summary (XAI)",
    "🔍 Forensic Audit & CBE Codes",
    "📊 Credit Risk & Financials",
    "🏛️ Institutional Quantitative Risk Lab",
    "💻 Raw Enriched JSON"
])

# -----------------------------------------------------------------------------
# TAB 1: Executive Summary & XAI
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# TAB 2: Forensic Audit & CBE Reason Codes
# -----------------------------------------------------------------------------
with tab2:
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
    st.subheader("Behavioral Banking Anomalies (Layer 2 Screening)")
    for anom in assessment["behavioral_anomalies"]:
        col_status, col_desc = st.columns([1, 4])
        with col_status:
            if anom["detected"]:
                st.markdown(f"🚨 **{anom['anomaly_name']}**")
            else:
                st.markdown(f"🟢 **{anom['anomaly_name']}**")
        with col_desc:
            st.write(anom["explanation_en"])

# -----------------------------------------------------------------------------
# TAB 3: Credit Risk & Financials
# -----------------------------------------------------------------------------
with tab3:
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

    st.markdown("#### Canonical Features Ready for Downstream Risk Model (Home Credit 413 Schema)")
    st.dataframe([combined_features], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Institutional Quantitative Risk Lab & Marginal Impact Engine
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🏛️ Institutional Quantitative Risk Lab & Portfolio Impact Engine")
    
    st.markdown("""
    <div class='disclaimer-badge'>⚠️ Model Governance & Regulatory Transparency Notice</div>
    <div class='callout-box'>
        <b>Model Governance & Prudential Architecture:</b><br>
        This quantitative cockpit dynamically bridges the <b>active individual applicant</b> with the bank's 
        <b>Reference Portfolio Baseline</b>. Marginal loss contributions (&Delta;ECL), IFRS 9 staging, and actuarial pricing 
        are computed live from the active payload.<br>
        <i>Note for Validators & Regulators:</i> Applicant Probability of Default (PD) in this prototype interface is 
        derived via calibrated heuristic business rules pending direct production API coupling with the trained ML models.
    </div>
    """, unsafe_allow_html=True)

    # Dynamic Applicant Risk Drivers
    is_returning = bool(payload.get("is_returning_customer", False))
    raw_salary = float(payload.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0.0) or 0.0)
    adjusted_salary = float(assessment["downstream_risk_feeder"]["risk_adjusted_salary"] or raw_salary)
    annuity = float(payload.get("form_data", {}).get("requested_annuity", 0.0) or 0.0)
    req_amount = float(payload.get("form_data", {}).get("requested_amount", 0.0) or 100000.0)
    iscore_score = float(payload.get("iscore_report_fields", {}).get("credit_score", {}).get("value", 650.0) or 650.0)
    
    # Dynamic PD proxy estimation
    base_calc_pd = 0.035 if is_returning else 0.085
    if iscore_score < 550:
        base_calc_pd += 0.065
    elif iscore_score > 720:
        base_calc_pd -= 0.015

    # IFRS 9 Staging & Loss Severity mapping
    if risk_level == "CRITICAL":
        app_pd = 1.00
        assigned_stage = "Stage 3 (Credit Impaired / Default)"
        stage_desc = "Fatal fraud / document tampering detected. Loan is non-performing upon origination."
        app_lgd = 0.55
    elif risk_level in ["MEDIUM", "HIGH"] or mismatch > 20.0:
        app_pd = min(0.35, base_calc_pd * 2.5)
        assigned_stage = "Stage 2 (Underperforming / SICR)"
        stage_desc = "Significant Increase in Credit Risk triggered via forensic mismatch or liquidity stress."
        app_lgd = 0.45
    else:
        app_pd = max(0.015, base_calc_pd)
        assigned_stage = "Stage 1 (Performing)"
        stage_desc = "Clean forensic audit and stable cashflows. Subject to 12-Month ECL provisioning."
        app_lgd = 0.45

    # Individual Marginal ECL Calculation
    app_ecl = req_amount * app_pd * app_lgd

    # Reference Portfolio Baseline Constants (Illustrative Egyptian Commercial Banking Book - 12,000 Facilities)
    BASE_EAD = 485.2        # EGP Millions
    BASE_ECL = 18.2         # EGP Millions
    BASE_NPL_RATE = 0.0768  # 7.68% Benchmark

    # Continuous Marginal Portfolio Deltas (Smooth Statistical Contribution)
    delta_ead_m = req_amount / 1e6
    delta_ecl_m = app_ecl / 1e6
    new_ead = BASE_EAD + delta_ead_m
    new_ecl = BASE_ECL + delta_ecl_m
    
    # Continuous Marginal Expected NPL Contribution
    base_npl_volume_m = BASE_EAD * BASE_NPL_RATE
    marginal_npl_addition_m = delta_ead_m * app_pd
    new_npl_rate = (base_npl_volume_m + marginal_npl_addition_m) / new_ead
    delta_npl_bps = (new_npl_rate - BASE_NPL_RATE) * 10000.0

    # Section 1: Live Marginal Impact KPI Cards
    st.markdown("#### 1. Live Applicant-to-Portfolio Marginal Risk Attribution")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Active Applicant Staging",
            value=assigned_stage.split(":")[0],
            delta="IFRS 9 Classification",
            delta_color="normal" if "Stage 1" in assigned_stage else "inverse"
        )
    with m2:
        st.metric(
            label="Applicant Estimated PD",
            value=f"{app_pd * 100:.1f}%",
            delta="Heuristic Policy Rule",
            delta_color="normal" if app_pd <= 0.15 else "inverse"
        )
    with m3:
        st.metric(
            label="Marginal ECL Provision Required",
            value=f"EGP {app_ecl:,.0f}",
            delta=f"+{delta_ecl_m*1000:.1f}k to Bank Reserves",
            delta_color="normal" if app_ecl < 10000 else "inverse"
        )
    with m4:
        st.metric(
            label="Post-Approval Portfolio NPL",
            value=f"{new_npl_rate * 100:.3f}%",
            delta=f"+{delta_npl_bps:.2f} bps" if delta_npl_bps > 0.001 else "Negligible Impact",
            delta_color="inverse" if delta_npl_bps > 0.5 else "normal"
        )

    st.caption(f"📌 **Active IFRS 9 Staging Rule:** {stage_desc}")
    st.caption(f"🏛️ **Reference Portfolio Baseline:** EGP {BASE_EAD:,.1f}M EAD | EGP {BASE_ECL:,.1f}M Baseline ECL | {BASE_NPL_RATE*100:.2f}% Baseline NPL.")
    st.markdown("---")

    # Section 2: Tailored Risk-Based Actuarial Pricing Engine
    st.markdown("#### 2. Tailored Risk-Based Actuarial Pricing Engine")
    st.write(f"Dynamic actuarial lending rate for **{applicant_name}** based on requested facility of **EGP {req_amount:,.0f}**:")

    # Cost of funds & capital building blocks
    cof = 18.0  # CBE reference corridor rate
    opex = 2.5  # Operational cost loading
    loss_margin = app_pd * app_lgd * 100
    raroc_hurdle = max(0.5, (1.0 - app_pd) * 3.5)
    total_suggested_rate = cof + opex + loss_margin + raroc_hurdle

    p_col1, p_col2 = st.columns([2, 1])
    with p_col1:
        pricing_breakdown = pd.DataFrame({
            "Pricing Building Block": [
                "1. Base Cost of Funds (CBE Benchmark)",
                "2. Operating Expense Loading (Origination & Servicing)",
                "3. Credit Expected Loss Margin (PD x LGD)",
                "4. Economic Capital Hurdle (RAROC 18% Target)",
                "Total Calculated Actuarial Lending Rate"
            ],
            "Percentage Rate (%)": [
                f"{cof:.2f}%",
                f"{opex:.2f}%",
                f"{loss_margin:.2f}% (Applicant Specific)",
                f"{raroc_hurdle:.2f}%",
                f"{total_suggested_rate:.2f}%" if risk_level != "CRITICAL" else "UNPRICEABLE (FATAL FRAUD)"
            ]
        })
        st.dataframe(pricing_breakdown, use_container_width=True)

    with p_col2:
        if risk_level == "CRITICAL":
            st.error("**Pricing Decision:**\n\n⛔ **UNPRICEABLE / REJECT**\n\nCannot price loan for critical fraud exposure.")
        else:
            st.success(f"**Recommended Lending Rate:**\n\n### **{total_suggested_rate:.2f}% per annum**\n\nAnnual Net Risk Spread: **+{raroc_hurdle:.2f}%**")

    st.markdown("---")

    # Section 3: Live Applicant Stress Testing (DTI Sensitivity)
    st.markdown("#### 3. Live Applicant Solvency Stress Testing (CBE Macro Shocks)")
    st.write("Evaluating how macroeconomic shocks impact **this specific applicant's** monthly debt-service capacity:")

    str_c1, str_c2 = st.columns([1, 2])
    with str_c1:
        st.markdown("**Simulate Macro Shocks:**")
        sim_hike = st.slider("CBE Corridor Rate Hike (+bps)", min_value=0, max_value=600, value=200, step=50)
        sim_cost_inflation = st.slider("Living Expense Inflation (+%)", min_value=0, max_value=25, value=10, step=5)

    with str_c2:
        # Dynamic DTI calculation
        current_dti = (annuity / adjusted_salary) if adjusted_salary > 0 else 1.0
        stressed_annuity = annuity * (1.0 + (sim_hike / 10000.0 * 2.2))
        stressed_income = max(1000.0, adjusted_salary * (1.0 - (sim_cost_inflation / 100.0 * 0.35)))
        stressed_dti = stressed_annuity / stressed_income

        cbe_limit = 0.50  # CBE 50% max DTI cap
        sd1, sd2, sd3 = st.columns(3)
        sd1.metric("Current Applicant DTI", f"{current_dti * 100:.1f}%", "Baseline")
        sd2.metric("Stressed DTI Ratio", f"{stressed_dti * 100:.1f}%", f"+{(stressed_dti - current_dti)*100:.1f}%", delta_color="inverse")
        sd3.metric("CBE Compliance Cap", "50.0%", "BREACHED" if stressed_dti > cbe_limit else "Compliant", delta_color="inverse" if stressed_dti > cbe_limit else "normal")

        if stressed_dti > cbe_limit:
            st.warning(f"⚠️ Under stress (+{sim_hike} bps hike), applicant's DTI breaches the CBE 50% legal threshold. Recommend reducing facility tenure or ticket size.")
        else:
            st.info(f"✅ Applicant remains solvent and compliant under stress scenario (+{sim_hike} bps rate hike).")

    st.markdown("---")

    # Section 4: Live Decile Placement
    st.markdown("#### 4. Active Applicant Decile Placement & Model Separation")
    st.write(f"Locating **{applicant_name}** (I-Score: {iscore_score:.0f}) on the bank's calibrated credit risk curve:")

    decile_data = {
        "Decile": [f"D{i}" for i in range(1, 11)],
        "Score Range": ["300-485", "486-540", "541-588", "589-630", "631-672", "673-710", "711-745", "746-780", "781-815", "816-850"],
        "Historical Bad Rate (%)": [38.5, 19.8, 11.8, 7.2, 4.3, 2.8, 1.5, 0.9, 0.5, 0.2],
        "Applicant Position": [""] * 10
    }
    if iscore_score < 486:
        active_dec = 0
    elif iscore_score < 541:
        active_dec = 1
    elif iscore_score < 589:
        active_dec = 2
    elif iscore_score < 631:
        active_dec = 3
    elif iscore_score < 673:
        active_dec = 4
    elif iscore_score < 711:
        active_dec = 5
    elif iscore_score < 746:
        active_dec = 6
    elif iscore_score < 781:
        active_dec = 7
    elif iscore_score < 816:
        active_dec = 8
    else:
        active_dec = 9

    decile_data["Applicant Position"][active_dec] = "🎯 ACTIVE APPLICANT HERE"
    d_df = pd.DataFrame(decile_data)
    st.dataframe(d_df, use_container_width=True)
    st.caption(f"Applicant falls into **{d_df.loc[active_dec, 'Decile']}** with an expected benchmark delinquency rate of **{d_df.loc[active_dec, 'Historical Bad Rate (%)']}%**.")

    st.markdown("---")

    # Section 5: Historical Empirical Segment Benchmarks
    st.markdown("#### 5. Historical Empirical Segment Benchmarks (Portfolio Track Record)")
    st.write(f"Benchmarking applicant tier against historical retail portfolio performance (**{'Returning Customer' if is_returning else 'New-to-Bank'}**):")

    seg1, seg2 = st.columns(2)
    with seg1:
        st.markdown("**Empirical Segment Observations:**")
        st.write(f"- **Active Segment Track:** {'Returning Bank Customer (65% Volume Share)' if is_returning else 'New-to-Bank Cold Start (35% Volume Share)'}")
        st.write(f"- **Historical Segment Default Rate:** {'5.82%' if is_returning else '11.45%'}")
        st.write(f"- **Discrimination Capacity (ROC-AUC):** {'0.792 (High Separation)' if is_returning else '0.738 (Moderate Separation)'}")
        st.write(f"- **5-Year Cumulative LTV:** {'EGP 48,200' if is_returning else 'EGP 21,500'}")
        st.caption("Empirical Conclusion: Maintaining returning customer relationships lowers baseline credit default probability by 49.2% relative to open-market cold-start acquisitions.")

    with seg2:
        seg_chart = pd.DataFrame({
            "Segment Category": ["Returning Customers", "New-to-Bank Applicants"],
            "12M Default Rate (%)": [5.82, 11.45],
            "Acquisition Cost / 100 (EGP)": [4.5, 18.5]
        }).set_index("Segment Category")
        st.bar_chart(seg_chart)

# -----------------------------------------------------------------------------
# TAB 5: Raw JSON Viewer
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Enriched Contract JSON Payload")
    enriched_payload = engine.enrich_payload(payload)
    st.json(enriched_payload)
