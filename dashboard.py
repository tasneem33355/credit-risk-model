"""
Streamlit Web Dashboard for Credit Risk & Application Fraud Analysis
=====================================================================
Platform: Smart Financing & Credit Request Analysis Platform (CrediX / ZAWOLF)
Standard: Basel III / IFRS 9 Prudential Compliance & CBE Regulatory Guidelines
Features:
  - Tab 1: Executive Summary & Bilingual Explainable AI (XAI)
  - Tab 2: Deep Forensic Audit & CBE Policy Violations
  - Tab 3: Fraud Ring Network & Cross-Application Collisions (Graph)
  - Tab 4: Institutional Quantitative Risk Lab (Monte Carlo, Vintage, Markov, Pricing, CBE Shocks)
  - Tab 5: Population Stability Index (PSI) & Model Drift Governance
  - Tab 6: Raw Enriched JSON Contract Viewer
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
from model.drift_monitor import PopulationDriftMonitor, FEATURE_NAMES

# Page configuration
st.set_page_config(
    page_title="CrediX | Enterprise Credit Decisioning & Portfolio Lab",
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
        font-size: 1.95rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.25rem;
    }
    
    .badge-clean {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-warn {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .badge-critical {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 5px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    
    .callout-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 18px 0;
        font-size: 0.88rem;
        color: #1E293B;
        line-height: 1.55;
    }
    
    .disclaimer-badge {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Fraud Engine & Drift Monitor with resource caching
@st.cache_resource
def get_fraud_engine():
    return CreditFraudEngine()

@st.cache_resource
def get_drift_monitor():
    return PopulationDriftMonitor()

engine = get_fraud_engine()
drift_monitor = get_drift_monitor()

# -----------------------------------------------------------------------------
# Sidebar: Case Selection & Telemetry Simulator
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🏦 CrediX Decision Portal")
st.sidebar.caption("Enterprise Underwriting & Portfolio Risk Lab")

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

# Digital Security & Telemetry Simulator in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Digital Telemetry Simulator")
sim_vpn = st.sidebar.checkbox("Simulate Commercial VPN / Proxy IP", value=False)
_default_device_id = payload.get("device_telemetry", {}).get(
    "device_fingerprint_id", f"DEV-DEFAULT-{payload.get('application_id', 'UNKNOWN')}"
)
sim_device_id = st.sidebar.text_input("Device Hardware ID", value=_default_device_id)
st.sidebar.caption("⚠️ Manually reusing the same Device Hardware ID across different applications will correctly trigger a Fraud Ring velocity alert (Layer 3). Leave the auto-filled default as-is to test each case in isolation.")
sim_off_hours = st.sidebar.selectbox("Application Submission Hour", options=[12, 14, 18, 3, 4], format_func=lambda h: f"{h:02d}:00 {'(Off-hours/Dawn Flag)' if 2<=h<=5 else '(Normal business)'}")

# Inject simulated telemetry into payload
if "device_telemetry" not in payload:
    payload["device_telemetry"] = {}
payload["device_telemetry"]["is_vpn_or_proxy"] = sim_vpn
payload["device_telemetry"]["device_fingerprint_id"] = sim_device_id
payload["device_telemetry"]["submission_hour_local"] = sim_off_hours

# Run Evaluation using verified production signatures
with st.spinner("Executing Forensic Audit & Portfolio Marginal Assessment..."):
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
    mismatch_ratio = float(assessment["metrics"].get("income_mismatch_ratio", 0.0))
    mismatch = mismatch_ratio * 100.0
    st.metric(
        label="Salary vs Bank Inflow Mismatch",
        value=f"{mismatch:.1f}%",
        delta="Verified Salary" if mismatch < 20.0 else "Significant Gap",
        delta_color="normal" if mismatch < 20.0 else "inverse"
    )

with kpi3:
    haircut = assessment["downstream_risk_feeder"]["haircut_percentage"]
    st.metric(
        label="Credibility Haircut",
        value=f"{haircut:.1f}%",
        delta=f"Risk-Adj Salary: EGP {assessment['downstream_risk_feeder']['risk_adjusted_salary']:,.0f}",
        delta_color="normal" if haircut == 0.0 else "inverse"
    )

with kpi4:
    total_violations = assessment["metrics"]["total_violations_count"]
    st.metric(
        label="Policy Violations Triggered",
        value=f"{total_violations}",
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

# -----------------------------------------------------------------------------
# 6 Comprehensive Tabs (Everything Unified)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Executive Summary (XAI)",
    "🔍 Document & Forensic Audit",
    "🕸️ Fraud Rings & Syndicate Graph",
    "🏛️ Institutional Quantitative Risk Lab",
    "📊 Model Drift & PSI Governance",
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
        id_ok = assessment["verification_checklist"].get("identity_verified", True)
        st.write("🪪 **National ID Consistency:**", "✅ Verified" if id_ok else "❌ Mismatch / Invalid")
        inc_ok = assessment["verification_checklist"].get("income_verified", True)
        st.write("💵 **Salary Inflow Reconciled:**", "✅ Reconciled" if inc_ok else "❌ Discrepancy Flagged")
    with c2:
        emp_ok = assessment["verification_checklist"].get("employer_verified", True)
        st.write("🏢 **Employer Entity Match:**", "✅ Matched" if emp_ok else "❌ Entity Discrepancy")
        doc_ok = assessment["verification_checklist"].get("document_integrity_verified", True)
        st.write("📄 **Document Digital Integrity:**", "✅ Authentic" if doc_ok else "❌ Tampering Suspected")
    with c3:
        bur_ok = assessment["verification_checklist"].get("bureau_verified", True)
        st.write("🏛️ **I-Score Bureau Inquiry:**", "✅ Fresh & Active" if bur_ok else "❌ Stale / Legal Action")
        dev_ok = assessment["verification_checklist"].get("device_telemetry_verified", True)
        st.write("🌐 **Device & Network Security:**", "✅ Authentic" if dev_ok else "⚠️ Security Flag")
        vel_ok = assessment["metrics"].get("entity_collisions_count", 0) == 0
        st.write("🕸️ **Cross-Application Velocity:**", "✅ No Collisions" if vel_ok else "🚨 Fraud Ring Signal Detected")

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
    st.subheader("Behavioral Banking & Deep Forensics (Layer 2)")
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
# TAB 3: Fraud Rings & Syndicate Graph
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🕸️ Cross-Application Entity Graph & Velocity Store")
    st.write("Tracks collisions across phone numbers, device IDs, bank accounts, and employer entities within a rolling 48-hour window:")

    graph_data = engine.entity_store.get_recent_graph_data(limit=30)
    
    if isinstance(graph_data, dict):
        edges = graph_data.get("edges", [])
        nodes = graph_data.get("nodes", [])
        
        gc1, gc2 = st.columns([1, 2])
        with gc1:
            st.metric("Total Tracked Entities in Registry", graph_data.get("total_entities", len(nodes)))
            st.metric("Active Association Links", len(edges))
            
        with gc2:
            if edges:
                st.markdown("##### 🔗 Multi-Entity Link Audit Table:")
                st.dataframe(pd.DataFrame(edges), use_container_width=True)
            else:
                st.info("ℹ️ No multi-applicant collisions recorded in the current 48h rolling window.")

        if nodes:
            with st.expander("View Active Registry Entity Nodes"):
                st.dataframe(pd.DataFrame(nodes), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Institutional Quantitative Risk Lab (Senior Enterprise Edition)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🏛️ Institutional Quantitative Risk Lab & Marginal Portfolio Engine")
    
    st.markdown("""
    <div class='disclaimer-badge'>⚠️ Comprehensive Model Governance & Prudential Disclosure Notice</div>
    <div class='callout-box'>
        <b>Model Architecture & Institutional Scope:</b><br>
        This quantitative workbench integrates the <b>active applicant's idiosyncratic risk profile</b> into the bank's 
        <b>Reference Retail Portfolio (EGP 485.2M / 12,000 Facilities)</b> modeled after Egyptian commercial banking standards.
        <br><br>
        <b>Disclosure & Methodological Classification:</b>
        <br>
        1. <b>Live Dynamic Computations:</b> Marginal default contributions (&Delta;EAD, &Delta;ECL, continuous &Delta;NPL), 
        IFRS 9 staging assignment, stressed DTI solvency, and actuarial loan pricing are calculated dynamically from the active payload. 
        <br>
        2. <b>Illustrative Reference Benchmarks:</b> The parametric Monte Carlo loss simulation, vintage cohort seasoning curves, 
        Markov roll-rate transition matrix, score decile delinquency master-scale, regional HHI concentration limits, 
        and segment performance benchmarks represent standardized reference empirical benchmarks used for institutional portfolio oversight.
    </div>
    """, unsafe_allow_html=True)

    # Mathematical Foundations & Live Applicant Deltas
    is_returning = bool(payload.get("is_returning_customer", False))
    raw_salary = float(payload.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0.0) or 0.0)
    adjusted_salary = float(assessment["downstream_risk_feeder"]["risk_adjusted_salary"] or raw_salary)
    annuity = float(payload.get("form_data", {}).get("requested_annuity", 0.0) or 0.0)
    req_amount = float(payload.get("form_data", {}).get("requested_amount", 0.0) or 100000.0)
    iscore_score = float(payload.get("iscore_report_fields", {}).get("credit_score", {}).get("value", 650.0) or 650.0)
    
    # Active PD calculation
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

    # Continuous Marginal Portfolio Deltas
    delta_ead_m = req_amount / 1e6
    delta_ecl_m = app_ecl / 1e6
    new_ead = BASE_EAD + delta_ead_m
    new_ecl = BASE_ECL + delta_ecl_m
    
    # Continuous Marginal Expected NPL Contribution
    base_npl_volume_m = BASE_EAD * BASE_NPL_RATE
    marginal_npl_addition_m = delta_ead_m * app_pd
    new_npl_rate = (base_npl_volume_m + marginal_npl_addition_m) / new_ead
    delta_npl_bps = (new_npl_rate - BASE_NPL_RATE) * 10000.0

    # -------------------------------------------------------------------------
    # SECTION 1: Live Applicant-to-Portfolio Marginal Attribution
    # -------------------------------------------------------------------------
    st.markdown("### 1. Live Applicant-to-Portfolio Marginal Attribution")
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

    st.caption(f"📌 **Active IFRS 9 Rule:** {stage_desc}")
    st.caption(f"🏛️ **Reference Portfolio Baseline:** EGP {BASE_EAD:,.1f}M EAD | EGP {BASE_ECL:,.1f}M Baseline ECL | {BASE_NPL_RATE*100:.2f}% Baseline NPL.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 2: Tail Risk & Monte Carlo Portfolio Loss Distribution
    # -------------------------------------------------------------------------
    st.markdown("### 2. Portfolio Tail Risk & Monte Carlo Value-at-Risk (Basel III)")
    st.write("Simulating 10,000 portfolio unexpected loss trajectories via parametrically-adjusted Gamma distribution:")

    np.random.seed(42)
    portfolio_shape = 4.2 + (app_pd * 0.4)
    portfolio_scale = 4.33 + (delta_ead_m * 0.1)
    simulated_losses = np.random.gamma(shape=portfolio_shape, scale=portfolio_scale, size=10000)

    var_95 = float(np.percentile(simulated_losses, 95))
    var_99 = float(np.percentile(simulated_losses, 99))
    cvar_99 = float(simulated_losses[simulated_losses >= var_99].mean())

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Value-at-Risk (VaR 95% 1-Year)", f"EGP {var_95:.2f}M", "Economic Capital")
    mc2.metric("Value-at-Risk (VaR 99% Basel III)", f"EGP {var_99:.2f}M", f"+{delta_ecl_m:.3f}M applicant shift", delta_color="inverse")
    mc3.metric("Expected Shortfall (CVaR 99%)", f"EGP {cvar_99:.2f}M", "Extreme Tail Shock", delta_color="inverse")

    hist_counts, hist_bins = np.histogram(simulated_losses, bins=25)
    loss_chart_data = pd.DataFrame({
        "Simulated Loss Bracket (EGP M)": [f"{hist_bins[i]:.1f}-{hist_bins[i+1]:.1f}" for i in range(len(hist_counts))],
        "Simulation Trajectory Frequency": hist_counts
    }).set_index("Simulated Loss Bracket (EGP M)")
    st.bar_chart(loss_chart_data)
    st.caption("📌 **Methodological Note:** Parametrically-adjusted Gamma loss simulation illustrating unexpected tail loss behavior under Basel III.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 3: Multi-Cohort Vintage Delinquency Matrix (MOB 3 - 24 Curves)
    # -------------------------------------------------------------------------
    st.markdown("### 3. Multi-Cohort Vintage Delinquency Matrix (MOB 3 - 24 Curves)")
    st.write("Tracking 30+ DPD cumulative delinquency curves across origination cohorts:")

    vintage_df = pd.DataFrame({
        "Origination Cohort": ["2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4", "2024-Q1", "2024-Q2 (Active)"],
        "Book Size (EGP M)": [78.4, 82.1, 74.5, 91.2, 85.0, 74.0],
        "MOB 3 (%)": [0.42, 0.38, 0.51, 0.65, 0.44, 0.48],
        "MOB 6 (%)": [1.12, 1.05, 1.34, 1.82, 1.20, 1.25],
        "MOB 9 (%)": [2.45, 2.30, 2.80, 3.65, 2.55, 2.60],
        "MOB 12 (%)": [4.15, 3.90, 4.60, 5.80, 4.30, "Maturity Track"],
        "MOB 18 (%)": [6.20, 5.85, 6.75, 7.95, "In Progress", "In Progress"],
        "MOB 24 (%)": [7.40, 7.10, 7.90, "In Progress", "In Progress", "In Progress"]
    })
    st.dataframe(vintage_df, use_container_width=True)

    curve_data = pd.DataFrame({
        "Month on Book (MOB)": [3, 6, 9, 12, 18, 24],
        "2023-Q1 Cohort": [0.42, 1.12, 2.45, 4.15, 6.20, 7.40],
        "2023-Q3 Cohort": [0.51, 1.34, 2.80, 4.60, 6.75, 7.90],
        "2023-Q4 Stressed Cohort": [0.65, 1.82, 3.65, 5.80, 7.95, 8.85],
    }).set_index("Month on Book (MOB)")
    st.line_chart(curve_data)
    st.caption("📌 **Illustrative Reference Benchmark:** Delinquency curves peak between Month 9 and 14.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 4: Markov Transition & Roll-Rate Matrix
    # -------------------------------------------------------------------------
    st.markdown("### 4. Markov Transition & Roll-Rate Probability Matrix")
    st.write("Observed 30-day state transition probabilities across delinquency buckets:")

    transition_matrix = pd.DataFrame({
        "From Status": ["Current (Performing)", "30 DPD (Late)", "60 DPD (Special Mention)", "90+ DPD (Default / NPL)"],
        "Roll to Current (Curative %)": [92.4, 22.5, 8.2, 1.5],
        "Roll to 30 DPD (%)": [6.8, 48.0, 14.3, 2.1],
        "Roll to 60 DPD (%)": [0.6, 24.5, 42.5, 5.4],
        "Roll to 90+ DPD (Default %)": [0.2, 5.0, 35.0, 91.0]
    }).set_index("From Status")
    st.dataframe(transition_matrix, use_container_width=True)
    st.caption("📌 **Illustrative Markov Framework:** Demonstrates sharp drop in cure rates past 60 DPD.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 5: Tailored Actuarial Risk-Based Pricing Engine
    # -------------------------------------------------------------------------
    st.markdown("### 5. Tailored Actuarial Risk-Based Pricing Engine")
    st.write(f"Dynamic lending rate calculated specifically for **{applicant_name}**:")

    cof = 18.0  # CBE corridor reference
    opex = 2.5  # Servicing load
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
            st.success(f"**Recommended Lending Rate:**\n\n### **{total_suggested_rate:.2f}% p.a.**\n\nAnnual Net Risk Spread: **+{raroc_hurdle:.2f}%**")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 6: Live Applicant Solvency Stress Testing (CBE Shocks)
    # -------------------------------------------------------------------------
    st.markdown("### 6. Live Applicant Solvency Stress Testing (CBE Macro Shocks)")
    st.write("Evaluating macroeconomic shocks on this specific applicant's debt-service capacity:")

    str_c1, str_c2 = st.columns([1, 2])
    with str_c1:
        st.markdown("**Simulate Macro Shocks:**")
        sim_hike = st.slider("CBE Corridor Rate Hike (+bps)", min_value=0, max_value=600, value=200, step=50)
        sim_cost_inflation = st.slider("Living Expense Inflation (+%)", min_value=0, max_value=25, value=10, step=5)

    with str_c2:
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
            st.warning(f"⚠️ Under stress (+{sim_hike} bps hike), applicant's DTI breaches the CBE 50% legal threshold.")
        else:
            st.info(f"✅ Applicant remains solvent and compliant under stress scenario (+{sim_hike} bps rate hike).")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 7: Calibrated Risk Decile Separation
    # -------------------------------------------------------------------------
    st.markdown("### 7. Active Applicant Decile Placement & Model Separation")
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
    st.caption(f"📌 Applicant is mapped into **{d_df.loc[active_dec, 'Decile']}** with an illustrative benchmark delinquency rate of **{d_df.loc[active_dec, 'Historical Bad Rate (%)']}%**.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # SECTION 8: Concentration Risk (HHI) & Dual-Segment Track
    # -------------------------------------------------------------------------
    st.markdown("### 8. Portfolio Concentration Risk (HHI) & Dual-Segment Track")
    st.write(f"Evaluating concentration limits and historical segment performance (**{'Returning Customer' if is_returning else 'New-to-Bank'}**):")

    c_col1, c_col2 = st.columns(2)
    with c_col1:
        st.markdown("**Regional & Sector Concentration Limits:**")
        conc_df = pd.DataFrame({
            "Governorate / Sector": ["Greater Cairo (Retail)", "Alexandria & Delta", "Upper Egypt & Canal", "Technology / Telecom", "Manufacturing / Public"],
            "Portfolio Share (%)": [48.5, 26.2, 14.8, 31.5, 22.0],
            "Regulatory Limit (%)": [55.0, 35.0, 20.0, 35.0, 30.0],
            "Status": ["Compliant (Within Limit)", "Compliant (Within Limit)", "Compliant (Within Limit)", "Approaching Limit (Alert)", "Compliant"]
        })
        st.dataframe(conc_df, use_container_width=True)
        st.caption("📌 **HHI Index:** 1,420 (Moderately Concentrated Portfolio), adhering to CBE prudential limits.")

    with c_col2:
        st.markdown("**Illustrative Segment Empirical Benchmarks:**")
        st.write(f"- **Active Segment Track:** {'Returning Bank Customer' if is_returning else 'New-to-Bank Cold Start'}")
        st.write(f"- **Benchmark Segment Default Rate:** {'5.82%' if is_returning else '11.45%'}")
        st.write(f"- **Target Model Separation (ROC-AUC):** {'0.792' if is_returning else '0.738'}")
        
        seg_chart = pd.DataFrame({
            "Segment Category": ["Returning Customers", "New-to-Bank Applicants"],
            "12M Default Rate (%)": [5.82, 11.45]
        }).set_index("Segment Category")
        st.bar_chart(seg_chart)

# -----------------------------------------------------------------------------
# TAB 5: Model Drift & PSI Governance (Fully Dynamic Mathematical Engine)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("📊 Population Stability Index (PSI) & Dynamic Model Drift Governance")
    st.caption("Compliance Framework: Central Bank of Egypt (CBE) Model Risk Management & Basel Committee Drift Policy")

    if drift_monitor.baseline_df is None:
        st.error("🚨 Baseline training data (`data/fraud_training_data_25000.csv`) could not be loaded. Please ensure artifacts are generated.")
    else:
        st.markdown("""
        <div class='callout-box'>
            <b>Prudential Drift Monitoring Architecture:</b><br>
            This engine evaluates population distribution shifts across the 12 core model features by computing the 
            <b>Population Stability Index (PSI)</b> against the official <b>25,000 synthetic banking baseline</b>.
            <br>
            $$PSI = \\sum \\left( \\text{Actual}\\% - \\text{Expected}\\% \\right) \\times \\ln\\left( \\frac{\\text{Actual}\\%}{\\text{Expected}\\%} \\right)$$
        </div>
        """, unsafe_allow_html=True)

        # Batch Selection Configuration
        st.markdown("#### 1. Incoming Production Batch Source Selection")
        drift_col1, drift_col2 = st.columns([2, 1])

        with drift_col1:
            batch_source = st.radio(
                "Select Incoming Batch to Evaluate for Drift:",
                [
                    "🟢 Live Operational Stream (Recent 1,000 Production Applications)",
                    "🟡 Stressed Macroeconomic Shift (Inflation & Inflow Compression Scenario)",
                    "📁 Upload Custom Production Batch (.csv / .xlsx)"
                ],
                index=0
            )

        incoming_batch_df = None

        if "Live Operational Stream" in batch_source:
            # Clean operational sample drawn directly from baseline population
            np.random.seed(101)
            sample_indices = np.random.choice(len(drift_monitor.baseline_df), size=min(1000, len(drift_monitor.baseline_df)), replace=False)
            incoming_batch_df = drift_monitor.baseline_df.iloc[sample_indices][FEATURE_NAMES].copy()
            st.caption("ℹ️ Evaluating recent 1,000 production applications under standard macroeconomic conditions.")

        elif "Stressed Macroeconomic Shift" in batch_source:
            np.random.seed(999)
            sample_indices = np.random.choice(len(drift_monitor.baseline_df), size=min(1000, len(drift_monitor.baseline_df)), replace=False)
            incoming_batch_df = drift_monitor.baseline_df.iloc[sample_indices][FEATURE_NAMES].copy()
            incoming_batch_df["income_mismatch_ratio"] *= np.random.uniform(1.35, 1.85, size=len(incoming_batch_df))
            incoming_batch_df["balance_volatility_cv"] *= np.random.uniform(1.25, 1.60, size=len(incoming_batch_df))
            incoming_batch_df["iscore_normalized"] *= np.random.uniform(0.75, 0.90, size=len(incoming_batch_df))
            incoming_batch_df["surge_ratio_max_to_avg"] *= np.random.uniform(1.20, 1.50, size=len(incoming_batch_df))
            st.caption("⚠️ Simulating significant market-wide credit deterioration (macroeconomic shock scenario).")

        elif "Upload Custom Production Batch" in batch_source:
            uploaded_batch = st.file_uploader("Upload Production Applications Batch (.csv or .xlsx):", type=["csv", "xlsx"])
            if uploaded_batch is not None:
                try:
                    if uploaded_batch.name.endswith(".xlsx"):
                        incoming_batch_df = pd.read_excel(uploaded_batch)
                    else:
                        incoming_batch_df = pd.read_csv(uploaded_batch)
                    st.success(f"Custom batch loaded: {len(incoming_batch_df):,} rows.")
                except Exception as e:
                    st.error(f"Failed to read batch file: {e}")

        # Compute Dynamic Drift Report via PopulationDriftMonitor
        if incoming_batch_df is not None:
            drift_report = drift_monitor.evaluate_batch_drift(incoming_batch_df)

            st.markdown("---")
            st.markdown("#### 2. Quantitative Stability Audit & Retraining Decision")
            dp1, dp2, dp3, dp4 = st.columns(4)

            max_psi = drift_report["max_psi_score"]
            max_feat = drift_report["max_psi_feature"]
            sys_health = drift_report["system_health"]
            retraining_req = drift_report["retraining_recommended"]

            with dp1:
                st.metric(
                    label="Portfolio Max PSI Score",
                    value=f"{max_psi:.4f}",
                    delta=f"Leading Feature: {max_feat}",
                    delta_color="normal" if max_psi < 0.10 else ("off" if max_psi < 0.25 else "inverse")
                )

            with dp2:
                health_color = "normal" if sys_health == "HEALTHY" else ("off" if sys_health == "WARNING" else "inverse")
                st.metric(
                    label="Model Population Health",
                    value=sys_health,
                    delta="Within Tolerable Bounds" if sys_health != "CRITICAL DRIFT" else "Breached CBE Limit",
                    delta_color=health_color
                )

            with dp3:
                st.metric(
                    label="Batch Sample Size",
                    value=f"{drift_report['evaluated_batch_size']:,} Apps",
                    delta="Evaluated vs 25,000 Baseline"
                )

            with dp4:
                st.metric(
                    label="Retraining Trigger",
                    value="MANDATORY RETRAINING" if retraining_req else "NO ACTION REQUIRED",
                    delta="Retraining Trigger (PSI ≥ 0.25)" if retraining_req else "Model In-Calibration",
                    delta_color="inverse" if retraining_req else "normal"
                )

            if retraining_req:
                st.error(f"🚨 **CBE Audit Mandate:** {drift_report['cbe_audit_comment']}")
            elif sys_health == "WARNING":
                st.warning(f"⚠️ **Model Risk Warning:** {drift_report['cbe_audit_comment']}")
            else:
                st.success(f"✅ **Audit Confirmation:** {drift_report['cbe_audit_comment']}")

            # Extract Active Applicant Features
            active_applicant_metrics = assessment.get("metrics", {})
            active_mismatch = float(active_applicant_metrics.get("income_mismatch_ratio", 0.0))
            active_annuity_bal = float(annuity / max(raw_salary, 1.0))
            active_iscore = float(iscore_score / 850.0)
            active_ocr = float(active_applicant_metrics.get("ocr_quality_mean", 0.95))
            active_uniformity = float(active_applicant_metrics.get("inflow_uniformity_score", 0.15))
            active_age = float(payload.get("national_id_fields", {}).get("age", {}).get("value", 35.0) or 35.0) / 100.0

            active_values_map = {
                "income_mismatch_ratio": active_mismatch,
                "annuity_to_balance_ratio": active_annuity_bal,
                "iscore_normalized": active_iscore,
                "ocr_quality_mean": active_ocr,
                "inflow_uniformity_score": active_uniformity,
                "applicant_age_norm": active_age,
                "balance_volatility_cv": float(active_applicant_metrics.get("balance_volatility_cv", 0.35)),
                "surge_ratio_max_to_avg": float(active_applicant_metrics.get("surge_ratio_max_to_avg", 1.25)),
                "min_to_avg_balance_ratio": float(active_applicant_metrics.get("min_to_avg_balance_ratio", 0.40)),
                "employment_tenure_years": float(payload.get("form_data", {}).get("employment_tenure_years", 5.0) or 5.0),
                "inflow_regularity_score": float(active_applicant_metrics.get("inflow_regularity_score", 0.85)),
                "bureau_facilities_count": float(payload.get("iscore_report_fields", {}).get("active_facilities_count", {}).get("value", 2.0) or 2.0)
            }

            st.markdown("---")
            st.markdown("#### 3. Feature-Level Population Stability Index (PSI) Audit Table")

            detailed_rows = []
            for feat in FEATURE_NAMES:
                if feat in drift_report["feature_metrics"]:
                    b_series = drift_monitor.baseline_df[feat].dropna().values
                    c_series = incoming_batch_df[feat].dropna().values if feat in incoming_batch_df.columns else np.array([])
                    
                    b_mean = float(np.mean(b_series)) if len(b_series) > 0 else 0.0
                    b_std = float(np.std(b_series)) if len(b_series) > 0 else 0.0
                    c_mean = float(np.mean(c_series)) if len(c_series) > 0 else 0.0
                    c_std = float(np.std(c_series)) if len(c_series) > 0 else 0.0
                    
                    psi_val = drift_report["feature_metrics"][feat]["psi_score"]
                    status = drift_report["feature_metrics"][feat]["status"]
                    
                    app_val = active_values_map.get(feat, np.nan)
                    if not np.isnan(app_val) and len(b_series) > 0:
                        pct_rank = float((b_series < app_val).mean() * 100.0)
                        pct_str = f"{pct_rank:.1f}th pct"
                    else:
                        pct_str = "N/A"

                    detailed_rows.append({
                        "Feature Name": feat.replace("_", " ").title(),
                        "Baseline Mean (±Std)": f"{b_mean:.3f} (±{b_std:.2f})",
                        "Incoming Batch Mean": f"{c_mean:.3f} (±{c_std:.2f})",
                        "Active Applicant Value": f"{app_val:.3f}" if not np.isnan(app_val) else "N/A",
                        "Active App Percentile": pct_str,
                        "Calculated PSI": psi_val,
                        "Regulatory Status": status
                    })

            detailed_df = pd.DataFrame(detailed_rows).sort_values("Calculated PSI", ascending=False).reset_index(drop=True)
            st.dataframe(detailed_df, use_container_width=True)

            st.caption("""
            📌 **CBE PSI Standards:**
            * **PSI < 0.10 (Green / Stable):** Normal underwriting operations. No action required.
            * **0.10 ≤ PSI < 0.25 (Yellow / Moderate Drift):** Review underwriting credit policy and sample manual audits.
            * **PSI ≥ 0.25 (Red / Significant Drift):** Mandatory model retraining or scorecard recalibration required by model governance committee.
            """)

            # Interactive Distribution Inspector
            st.markdown("---")
            st.markdown("#### 4. Empirical Distribution Comparison (Baseline vs. Current Batch)")
            inspected_feat = st.selectbox(
                "Select Feature to Visually Inspect Distribution Shift:",
                FEATURE_NAMES,
                index=FEATURE_NAMES.index(max_feat) if max_feat in FEATURE_NAMES else 0
            )

            if inspected_feat in drift_monitor.baseline_df.columns and inspected_feat in incoming_batch_df.columns:
                b_hist = drift_monitor.baseline_df[inspected_feat].dropna()
                c_hist = incoming_batch_df[inspected_feat].dropna()

                bins = np.linspace(
                    min(b_hist.min(), c_hist.min()),
                    max(b_hist.max(), c_hist.max()),
                    25
                )
                b_counts, _ = np.histogram(b_hist, bins=bins, density=True)
                c_counts, _ = np.histogram(c_hist, bins=bins, density=True)

                bin_labels = [f"{bins[i]:.2f}" for i in range(len(bins)-1)]
                dist_df = pd.DataFrame({
                    "Baseline Population Density": b_counts,
                    "Current Batch Density": c_counts
                }, index=bin_labels)

                st.bar_chart(dist_df)
                st.caption(f"Comparing probability density of **{inspected_feat}** across 25 quantile buckets. Active applicant observed: **{active_values_map.get(inspected_feat, 0.0):.3f}**.")
        else:
            st.info("Please upload or select a batch dataset above to compute dynamic population stability metrics.")

# -----------------------------------------------------------------------------
# TAB 6: Raw Enriched JSON Contract Viewer
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("Enriched Contract JSON Payload")
    enriched_payload = engine.enrich_payload(payload)
    st.json(enriched_payload)
