"""
Application Data Contract (v2) to Model Feature Vector Adapter
===============================================================
Maps incoming Egyptian Application JSON (OCR + Bureau + Core Banking)
into the feature representation expected by the Credit Risk XGBoost/LightGBM model.
"""

from typing import Dict, Any, Tuple


def adapt_application_to_model_inputs(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Transforms an Egyptian Credit Application JSON (v2) into:
      1. `application`: Dict of raw application-level fields (Home Credit naming)
      2. `history_features`: Dict of pre-computed aggregates (PREV_*, INST_*, POS_*, CC_*)
    """
    form = payload.get("form_data", {})
    nid = payload.get("national_id_fields", {})
    salary = payload.get("salary_certificate_fields", {})
    bank = payload.get("bank_statement_fields", {})
    iscore = payload.get("iscore_report_fields", {})
    history = payload.get("internal_history", {})
    agg = history.get("aggregated_metrics", {})
    
    # Check if there is a risk-adjusted salary from fraud engine, else use declared
    fraud_feeder = payload.get("fraud_assessment", {}).get("downstream_risk_feeder", {})
    declared_net = float(salary.get("declared_net_salary", {}).get("value", 0.0) or 0.0)
    monthly_income = fraud_feeder.get("risk_adjusted_salary", declared_net)
    
    # Calculate age in days
    age_years = float(nid.get("age_years", {}).get("value", 35.0) or 35.0)
    days_birth = int(-round(age_years * 365.25))
    
    # Employment tenure in days
    tenure_years = float(salary.get("employment_tenure_years", {}).get("value", 3.0) or 3.0)
    days_employed = int(-round(tenure_years * 365.25))
    
    # Credit bureau proxy from I-Score (300-850 mapped to [0, 1])
    credit_score_raw = iscore.get("credit_score", {}).get("value")
    if credit_score_raw is not None:
        ext_source_proxy = max(0.0, min(1.0, (float(credit_score_raw) - 300.0) / 550.0))
    else:
        ext_source_proxy = 0.50

    # 1. Base Application Dictionary
    app_dict = {
        "AMT_CREDIT": float(form.get("requested_amount", 100000.0)),
        "AMT_ANNUITY": float(form.get("requested_annuity", 4000.0)),
        "AMT_INCOME_TOTAL": monthly_income * 12.0,  # Annualized
        "AMT_GOODS_PRICE": float(form.get("goods_price", 0.0) or form.get("requested_amount", 100000.0)),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "CNT_CHILDREN": int(form.get("children_count", 0)),
        "CNT_FAM_MEMBERS": int(form.get("family_members_count", 1)),
        "FLAG_OWN_CAR": 1 if form.get("owns_car", False) else 0,
        "FLAG_OWN_REALTY": 1 if form.get("owns_realty", False) else 0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": str(nid.get("gender", {}).get("value", "M")),
        "NAME_FAMILY_STATUS": str(form.get("family_status", "Married")).capitalize(),
        "NAME_HOUSING_TYPE": "House / apartment" if form.get("housing_type") == "owned" else "Rented apartment",
        "NAME_EDUCATION_TYPE": "Higher education" if form.get("education_type") == "higher_education" else "Secondary / secondary special",
        "NAME_INCOME_TYPE": "Working",
        "OCCUPATION_TYPE": str(salary.get("job_title", {}).get("value", "Core staff")),
        "ORGANIZATION_TYPE": str(salary.get("employer_sector", {}).get("value", "Business Entity Type 3")),
        # I-Score proxies for external sources
        "EXT_SOURCE_1": ext_source_proxy,
        "EXT_SOURCE_2": ext_source_proxy,
        "EXT_SOURCE_3": ext_source_proxy,
    }

    # 2. History Features (Core Banking + I-Score aggregates)
    history_missing = int(history.get("internal_history_missing", 1))
    
    # Bureau aggregate mappings from I-Score
    bureau_active_loans = len(iscore.get("bureau_facilities", []))
    bureau_total_debt = float(iscore.get("total_outstanding_balance", {}).get("value", 0.0) or 0.0)
    bureau_total_credit = float(iscore.get("total_active_loans_limit", {}).get("value", 0.0) or 0.0)
    bureau_max_dpd = int(iscore.get("max_days_past_due", {}).get("value", 0) or 0)

    history_features = {
        "INTERNAL_HISTORY_MISSING": float(history_missing),
        # Bureau aggregates (external)
        "BUREAU_LOAN_COUNT": float(bureau_active_loans),
        "BUREAU_TOTAL_DEBT": bureau_total_debt,
        "BUREAU_TOTAL_CREDIT": bureau_total_credit,
        "BUREAU_MAX_OVERDUE": float(bureau_max_dpd),
        # Internal bank history (populated if returning customer, 0 if new-to-bank)
        "PREV_APP_COUNT": float(agg.get("prev_app_count", 0.0)),
        "PREV_APPROVED_RATIO": float(agg.get("prev_approved_ratio", 0.0)),
        "PREV_AVG_CREDIT": float(agg.get("prev_avg_credit", 0.0)),
        "INST_PAYMENT_COUNT": float(agg.get("inst_payment_count", 0.0)),
        "INST_LATE_COUNT": float(agg.get("inst_late_count", 0.0)),
        "INST_SEVERE_LATE_COUNT": float(agg.get("inst_severe_late_count", 0.0)),
        "INST_LATE_RATIO": float(agg.get("inst_late_ratio", 0.0)),
        "INST_AVG_DAYS_LATE": float(agg.get("inst_avg_days_late", 0.0)),
        "INST_MAX_DAYS_LATE": float(agg.get("inst_max_days_late", 0.0)),
        "POS_RECORD_COUNT": float(agg.get("pos_record_count", 0.0)),
        "POS_AVG_DPD": float(agg.get("pos_avg_dpd", 0.0)),
        "CC_AVG_BALANCE": float(agg.get("cc_avg_balance", 0.0)),
        "CC_BALANCE_TO_LIMIT_RATIO": float(agg.get("cc_balance_to_limit_ratio", 0.0)),
    }

    return app_dict, history_features
