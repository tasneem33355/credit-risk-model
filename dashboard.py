"""
Streamlit Web Dashboard for Credit Risk & Application Fraud Analysis
=====================================================================
Platform: Smart Financing & Credit Request Analysis Platform (ZAWOLF)
Author: AI Credit & Risk Analytics Team
"""

import streamlit as st
import json
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs

# Page configuration
st.set_page_config(
    page_title="ZAWOLF | منصة التحليل الائتماني وكشف الاحتيال الذكي",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling and RTL support
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-yellow {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    .kpi-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Engine
@st.cache_resource
def get_fraud_engine():
    return CreditFraudEngine()

engine = get_fraud_engine()

# Sidebar: Case Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=70)
st.sidebar.title("لوحة تحكم مسؤول الائتمان")
st.sidebar.markdown("**Credit & Fraud Decisioning Center**")

preset_options = {
    "1. عميل بنك حالي (منضبط) - Returning Customer": "sample_returning_customer_payload.json",
    "2. عميل جديد للبنك (Cold Start) - New to Bank": "sample_new_to_bank_payload.json",
    "3. حالة احتيال وتزوير مستندات (Fraud Injected)": "sample_fraudulent_applicant_payload.json",
    "4. رفع ملف JSON مخصص (Custom Upload)": "custom"
}

selected_option = st.sidebar.selectbox("اختر ملف الطلب للاختبار:", list(preset_options.keys()))

payload = None
filename = preset_options[selected_option]

if filename == "custom":
    uploaded_file = st.sidebar.file_uploader("ارفع ملف JSON مطابق لعقد البيانات:", type=["json"])
    if uploaded_file is not None:
        try:
            payload = json.load(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"خطأ في قراءة ملف الـ JSON: {e}")
else:
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        st.sidebar.warning(f"الملف {filename} غير موجود بالفولدر.")

if payload is None:
    st.info("👈 برجاء اختيار أو رفع ملف طلب تمويل من القائمة الجانبية لبدء التقييم.")
    st.stop()

# Run Evaluation
with st.spinner("جاري تشغيل محرك الفحص الجنائي والمتقاطع..."):
    assessment = engine.evaluate(payload)
    app_fe, hist_fe = adapt_application_to_model_inputs(payload)

# Header Section
app_id = payload.get("application_id", "N/A")
applicant_name = payload.get("national_id_fields", {}).get("full_name", {}).get("value", "عميل غير محدد")
loan_purpose = payload.get("form_data", {}).get("loan_purpose", "personal_cash")
requested_amount = float(payload.get("form_data", {}).get("requested_amount", 0.0))

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"<div class='main-header'>طلب التمويل: {app_id}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-header'>مقدم الطلب: <b>{applicant_name}</b> | الغرض: <b>{loan_purpose}</b> | المبلغ المطلوب: <b>{requested_amount:,.0f} ج.م</b></div>", unsafe_allow_html=True)

with col_status:
    risk_level = assessment["fraud_risk_level"]
    if risk_level == "LOW":
        st.markdown("<div class='badge-green'>✅ مستندات سليمة (Low Fraud Risk)</div>", unsafe_allow_html=True)
    elif risk_level in ["MEDIUM", "HIGH"]:
        st.markdown(f"<div class='badge-yellow'>⚠️ اشتباه وتضارب ({risk_level})</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-red'>🚨 تزوير عالي الخطورة (CRITICAL)</div>", unsafe_allow_html=True)

st.markdown("---")

# KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        label="سكور الاحتيال (Fraud Score)",
        value=f"{assessment['fraud_risk_score']:.2f} / 1.00",
        delta="خطر منخفض" if assessment['fraud_risk_score'] <= 0.25 else "- مؤشر خطر",
        delta_color="normal" if assessment['fraud_risk_score'] <= 0.25 else "inverse"
    )

with kpi2:
    mismatch = assessment["metrics"]["income_mismatch_ratio"] * 100
    st.metric(
        label="تضارب الراتب مع كشف الحساب",
        value=f"{mismatch:.1f}%",
        delta="مطابق تماماً" if mismatch <= 10 else f"فرق {mismatch:.0f}%",
        delta_color="normal" if mismatch <= 10 else "inverse"
    )

with kpi3:
    haircut = assessment["downstream_risk_feeder"]["haircut_percentage"]
    adj_sal = assessment["downstream_risk_feeder"]["risk_adjusted_salary"]
    st.metric(
        label="الراتب المعتمد للائتمان",
        value=f"{adj_sal:,.0f} ج.م",
        delta=f"خصم {haircut}% مصداقية" if haircut > 0 else "معتمد 100%",
        delta_color="normal" if haircut == 0 else "inverse"
    )

with kpi4:
    st.metric(
        label="مخالفات الأدلة الجنائية",
        value=f"{assessment['metrics']['total_violations_count']} مخالفة",
        delta=f"{assessment['metrics']['critical_violations_count']} جسيمة" if assessment['metrics']['critical_violations_count'] > 0 else "نظيف",
        delta_color="normal" if assessment['metrics']['critical_violations_count'] == 0 else "inverse"
    )

# Executive Recommendation Banner
action_box = st.container()
if risk_level == "LOW":
    action_box.success(f"**القرار المقترح:** {assessment['action_ar']} (`{assessment['recommended_action']}`)")
elif risk_level in ["MEDIUM", "HIGH"]:
    action_box.warning(f"**القرار المقترح:** {assessment['action_ar']} (`{assessment['recommended_action']}`)")
else:
    action_box.error(f"**القرار المقترح:** {assessment['action_ar']} (`{assessment['recommended_action']}`)")

# Detailed Tabs
tab_exec, tab_fraud, tab_credit, tab_json = st.tabs([
    "📋 ملخص التقرير التنفيذي (XAI)",
    "🔍 الفحص الجنائي والمستندي (Forensics)",
    "📊 الملف الائتماني والريسك (Credit Risk)",
    "💻 مستعرض الـ JSON الكامل (Payload)"
])

# -----------------------------------------------------------------------------
# TAB 1: Executive XAI Summary
# -----------------------------------------------------------------------------
with tab_exec:
    st.subheader("تقرير الذكاء الاصطناعي التفسيري لمسؤول الائتمان (Explainable AI Summary)")
    st.info(assessment["explainable_ai"]["executive_summary_ar"])
    st.markdown(f"**English Narrative:** {assessment['explainable_ai']['executive_summary_en']}")
    
    st.markdown("### قائمة الفحوصات المتقاطعة (Cross-Document Verification Checklist)")
    c1, c2, c3 = st.columns(3)
    with c1:
        id_ok = assessment["verification_checklist"]["identity_verified"]
        st.write("🪪 **التحقق من الهوية والرقم القومي:**", "✅ متطابق" if id_ok else "❌ غير متطابق")
        inc_ok = assessment["verification_checklist"]["income_verified"]
        st.write("💵 **التحقق من صحة الدخل:**", "✅ متطابق" if inc_ok else "❌ تضارب غير مبرر")
    with c2:
        emp_ok = assessment["verification_checklist"]["employer_verified"]
        st.write("🏢 **التحقق من جهة العمل:**", "✅ متطابقة" if emp_ok else "❌ جهة العمل مختلفة")
        doc_ok = assessment["verification_checklist"]["document_integrity_verified"]
        st.write("📑 **سلامة المستندات من التعديل:**", "✅ أصلية" if doc_ok else "❌ اشتباه تلاعب رقمي")
    with c3:
        bur_ok = assessment["verification_checklist"]["bureau_verified"]
        st.write("🏛️ **الاستعلام الائتماني (I-Score):**", "✅ سارٍ" if bur_ok else "❌ متأخر أو منتهٍ")
        math_ok = assessment["verification_checklist"]["bank_statement_math_verified"]
        st.write("🧮 **توازن حسابات كشف الحساب:**", "✅ سليم" if math_ok else "❌ خلل حسابي")

# -----------------------------------------------------------------------------
# TAB 2: Document Forensics
# -----------------------------------------------------------------------------
with tab_fraud:
    st.subheader("تفاصيل المخالفات وأكواد البنك المركزي (CBE Regulatory Reason Codes)")
    
    if assessment["triggered_rules"]:
        for rule in assessment["triggered_rules"]:
            with st.expander(f"[{rule['severity']}] {rule['rule_name_ar']} ({rule['rule_code']})", expanded=True):
                st.write(f"**الوصف بالعربية:** {rule['description_ar']}")
                st.write(f"**English Description:** {rule['description_en']}")
                col_obs, col_thresh = st.columns(2)
                col_obs.caption(f"القيمة المرصودة: `{rule['observed_value']}`")
                col_thresh.caption(f"الحد المسموح به: `{rule['threshold_value']}`")
    else:
        st.success("✅ لم يتم تسجيل أي مخالفة أو خرق لقواعد التطابق الجنائي والمستندي.")

    st.markdown("---")
    st.subheader("السلوكيات المصرفية الشاذة (Behavioral Anomalies)")
    for anom in assessment["behavioral_anomalies"]:
        col_a_status, col_a_desc = st.columns([1, 4])
        with col_a_status:
            if anom["detected"]:
                st.markdown(f"🚨 **{anom['anomaly_name']}**")
            else:
                st.markdown(f"🟢 {anom['anomaly_name']}")
        with col_a_desc:
            st.write(anom["explanation_ar"])

# -----------------------------------------------------------------------------
# TAB 3: Credit Risk & Demographics
# -----------------------------------------------------------------------------
with tab_credit:
    st.subheader("الملف الائتماني والبيانات المهنية (Credit & Demographic Profile)")
    
    col_cr1, col_cr2 = st.columns(2)
    with col_cr1:
        st.markdown("#### بيانات العميل والدخل")
        dec_salary = payload.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0)
        st.write(f"- **صافي الراتب المصرح به:** {dec_salary:,.0f} ج.م / شهرياً")
        st.write(f"- **الراتب المعتمد بعد الفحص:** {assessment['downstream_risk_feeder']['risk_adjusted_salary']:,.0f} ج.م")
        st.write(f"- **السن:** {payload.get('national_id_fields', {}).get('age_years', {}).get('value', 'N/A')} سنة")
        st.write(f"- **جهة العمل:** {payload.get('salary_certificate_fields', {}).get('employer_name', {}).get('value', 'N/A')}")
        st.write(f"- **مدة الخدمة:** {payload.get('salary_certificate_fields', {}).get('employment_tenure_years', {}).get('value', 'N/A')} سنة")
        
    with col_cr2:
        st.markdown("#### التقرير الائتماني (I-Score)")
        iscore_val = payload.get("iscore_report_fields", {}).get("credit_score", {}).get("value", "غير متوفر")
        st.write(f"- **Credit Score:** {iscore_val}")
        st.write(f"- **حالة العميل بالبنك:** {'عميل حالي (Returning)' if payload.get('is_returning_customer') else 'عميل جديد كلياً (New to Bank)'}")
        st.write(f"- **القسط الشهري المطلوب:** {payload.get('form_data', {}).get('requested_annuity', 0):,.0f} ج.م")
        st.write(f"- **مدة السداد المقترحة:** {payload.get('form_data', {}).get('tenure_months', 0)} شهراً")

    st.markdown("#### الفيتشرز المجهزة للإطعام في موديل الـ Risk (Model Ready Features)")
    st.dataframe([app_fe], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: JSON Viewer
# -----------------------------------------------------------------------------
with tab_json:
    st.subheader("مستعرض ملف الـ JSON المخصب والنهائي")
    enriched_payload = engine.enrich_payload(payload)
    st.json(enriched_payload)
