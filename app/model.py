"""
Inference layer: loads the trained artifacts, turns an incoming request
into the exact feature vector the model was trained on, and converts the
predicted Probability of Default (PD) into a credit score, decision, and
adverse-action reason codes.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from . import config

# Make model/features.py importable without turning model/ into a package,
# so training code and serving code stay in sync.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from features import add_core_features, add_secondary_features  # noqa: E402

MODEL_VERSION = "credit-risk-xgb-lgb-blend-v1"


class ModelBundle:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "xgb_final.pkl"), "rb") as f:
            self.xgb_model = pickle.load(f)
        with open(os.path.join(model_dir, "lgb_final.pkl"), "rb") as f:
            self.lgb_model = pickle.load(f)
        with open(os.path.join(model_dir, "encoder.pkl"), "rb") as f:
            self.encoder = pickle.load(f)
        with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
            self.feature_names: List[str] = pickle.load(f)
        with open(os.path.join(model_dir, "feature_schema.json")) as f:
            self.schema: Dict[str, Any] = json.load(f)

        metrics_path = os.path.join(model_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)
            self.w_xgb = self.metrics.get("blend_weights", {}).get("xgb", config.DEFAULT_W_XGB)
            self.w_lgb = self.metrics.get("blend_weights", {}).get("lgb", config.DEFAULT_W_LGB)
        else:
            self.metrics = {}
            self.w_xgb, self.w_lgb = config.DEFAULT_W_XGB, config.DEFAULT_W_LGB


@lru_cache(maxsize=1)
def load_models() -> ModelBundle:
    return ModelBundle(config.MODEL_DIR)


def build_feature_vector(application: Dict[str, Any], history_features: Dict[str, float] | None, bundle: ModelBundle) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Turn one raw application (plus optional precomputed history
    features) into the ordered sparse feature vector the ensemble expects.

    Any expected feature not derivable from the request is filled with 0.
    This is exact for the application-level block (core + secondary FE,
    one-hot encoded with the fitted training encoder) and is a documented
    approximation for the bureau / previous-loan / installments / POS /
    credit-card blocks unless `history_features` supplies them.
    """
    row = pd.DataFrame([application])

    # Application-level feature engineering, reusing the exact training logic.
    # add_core_features/add_secondary_features expect the standard Home
    # Credit column set; run best-effort and fill anything still missing.
    try:
        row = add_core_features(row)
        row = add_secondary_features(row)
    except KeyError as exc:
        raise ValueError(f"Missing required application field: {exc}") from exc

    numeric_cols = bundle.schema["application_numeric_cols"]
    categorical_cols = bundle.schema["application_categorical_cols"]

    X_num = row.reindex(columns=numeric_cols, fill_value=0)
    X_num = X_num.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    X_cat = row.reindex(columns=categorical_cols, fill_value="Missing").fillna("Missing").astype(str)
    X_cat_encoded = bundle.encoder.transform(X_cat)

    X_pos = sparse.hstack(
        [sparse.csr_matrix(X_num.values, dtype=np.float32), X_cat_encoded], format="csr", dtype=np.float32
    )
    app_feature_names = numeric_cols + bundle.encoder.get_feature_names_out(categorical_cols).tolist()

    # Assemble the full 413-wide vector in the exact training order,
    # overriding zeros with any supplied history features.
    full = pd.Series(0.0, index=bundle.feature_names, dtype=np.float32)
    app_series = pd.Series(np.asarray(X_pos.todense()).ravel(), index=app_feature_names)
    full.update(app_series.reindex(full.index).dropna())

    used_history_defaults = True
    if history_features:
        overlap = {k: v for k, v in history_features.items() if k in full.index}
        if overlap:
            full.update(pd.Series(overlap, dtype=np.float32))
            used_history_defaults = False

    X_final = sparse.csr_matrix(full.values.reshape(1, -1).astype(np.float32))
    debug_info = {"used_history_defaults": used_history_defaults}
    return X_final, debug_info


def predict_pd(X: sparse.csr_matrix, bundle: ModelBundle) -> float:
    prob_xgb = bundle.xgb_model.predict_proba(X)[:, 1][0]
    prob_lgb = bundle.lgb_model.predict_proba(X)[:, 1][0]
    return float(bundle.w_xgb * prob_xgb + bundle.w_lgb * prob_lgb)


def compute_credit_score(prob: float) -> int:
    score = config.CREDIT_SCORE_MAX - (prob * (config.CREDIT_SCORE_MAX - config.CREDIT_SCORE_MIN))
    return int(np.clip(round(score), config.CREDIT_SCORE_MIN, config.CREDIT_SCORE_MAX))


def decide(prob_default: float) -> Tuple[str, str]:
    if prob_default < config.CUTOFF_APPROVE:
        return "AUTO-APPROVE", "Low Risk (Grade A/B)"
    elif prob_default < config.CUTOFF_REJECT:
        return "MANUAL REVIEW", "Medium Risk (Grade C/D) - Request Collateral/Guarantor"
    else:
        return "AUTO-REJECT", "High Risk (Grade E) - Decline Application"


def compute_business_impact(prob_default: float, amt_credit: float) -> Dict[str, Any]:
    """Per-applicant translation of the docs/model_card.md §4 P&L assumptions
    (NIM = 10% of AMT_CREDIT on performing loans, LGD = 45% of AMT_CREDIT on
    default) -- the bank's own business framing, applied to one applicant
    instead of the portfolio-level simulation."""
    lgd = config.LOSS_GIVEN_DEFAULT
    nim = config.NET_INTEREST_MARGIN

    expected_loss = prob_default * lgd * amt_credit
    expected_annual_profit = nim * amt_credit
    net_expected_value = expected_annual_profit - expected_loss
    risk_adjusted_return = (net_expected_value / amt_credit) if amt_credit else 0.0

    return {
        "requested_amount": round(amt_credit, 2),
        "expected_loss_if_default": round(expected_loss, 2),
        "expected_annual_profit_if_performing": round(expected_annual_profit, 2),
        "net_expected_value": round(net_expected_value, 2),
        "risk_adjusted_return": f"{risk_adjusted_return*100:.2f}%",
        "assumptions": {"loss_given_default": lgd, "net_interest_margin": nim},
    }


def portfolio_context(decision: str, prob_default: float) -> Dict[str, Any]:
    """Frame this applicant's decision against the §3 portfolio simulation
    table in docs/model_card.md -- i.e. what NPL rate a bank running this
    model at a similar acceptance posture has historically carried, instead
    of comparing this one PD to an arbitrary single number."""
    market = config.MARKET_BAD_RATE
    table = config.PORTFOLIO_NPL_BY_ACCEPTANCE

    if decision == "AUTO-APPROVE":
        npl_low, npl_high = table[0.40], table[0.70]
        note = (
            f"Applicant falls in the low-risk band the model card's portfolio "
            f"simulation (§3) associates with acceptance rates of 40-70%, where "
            f"historical portfolio NPL runs {npl_low*100:.2f}%-{npl_high*100:.2f}% "
            f"vs. a {market*100:.2f}% market baseline."
        )
    elif decision == "MANUAL REVIEW":
        npl_mid = table[0.80]
        note = (
            f"Applicant sits between the auto-approve and auto-reject cutoffs "
            f"(§5) -- comparable to the {npl_mid*100:.2f}% portfolio NPL band at "
            f"~80% acceptance in §3. Collateral/guarantor terms are the model "
            f"card's recommended treatment at this risk level, not an outright "
            f"decline."
        )
    else:
        note = (
            f"Applicant's PD ({prob_default*100:.2f}%) is above the §3 "
            f"portfolio simulation's highest-NPL band "
            f"({table[0.85]*100:.2f}% at 85% acceptance) -- accepting "
            f"applicants at this risk level has historically raised portfolio "
            f"NPL above the {market*100:.2f}% market baseline rather than "
            f"reducing it (see §4 P&L simulation)."
        )
    return {"framing": note}

def reason_codes(application: Dict[str, Any], full_features: Dict[str, float]) -> List[str]:
    """Rule-based adverse-action reason codes, required under fair-lending
    regulation whenever an applicant is rejected or referred for review."""
    codes = []

    if full_features.get("EXT_SOURCE_MEAN", 1.0) < 0.35:
        codes.append("Low External Bureau Score / Credit History Rating")
    if full_features.get("INST_LATE_RATIO", 0.0) > 0.15:
        codes.append("Historical Late Payment Record on Past Loans")
    if full_features.get("CREDIT_INCOME_RATIO", 0.0) > 4.0:
        codes.append("High Credit-to-Income Ratio (Excessive Leverage)")
    if full_features.get("CC_AVG_UTILIZATION", 0.0) > 0.60:
        codes.append("High Revolving Credit Card Balance Utilization")
    if full_features.get("PREV_REFUSED_RATIO", 0.0) > 0.25:
        codes.append("High Previous Loan Rejection History")
    if full_features.get("EMPLOYED_YEARS", 10.0) < 1.0:
        codes.append("Short Employment Tenure (< 1 Year)")

    if not codes:
        codes.append("No critical risk flags detected; standard portfolio profile")
    return codes[:3]


def score_application(application: Dict[str, Any], history_features: Dict[str, float] | None = None) -> Dict[str, Any]:
    bundle = load_models()
    X, debug_info = build_feature_vector(application, history_features, bundle)
    prob_default = predict_pd(X, bundle)

    credit_score = compute_credit_score(prob_default)
    decision, risk_tier = decide(prob_default)

    full_features = dict(zip(bundle.feature_names, np.asarray(X.todense()).ravel().tolist()))
    codes = reason_codes(application, full_features)
    amt_credit = float(application.get("AMT_CREDIT", 0) or 0)
    business_impact = compute_business_impact(prob_default, amt_credit)
    portfolio_note = portfolio_context(decision, prob_default)
    
    return {
        "credit_score": credit_score,
        "probability_of_default": f"{prob_default*100:.2f}%",
        "decision": decision,
        "risk_tier": risk_tier,
        "reason_codes": codes,
        "business_impact": business_impact,
        "portfolio_context": portfolio_note,
        "model_version": MODEL_VERSION,
        "used_history_defaults": debug_info["used_history_defaults"],
    }
