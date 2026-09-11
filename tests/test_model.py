import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import model, config


def test_compute_credit_score_bounds():
    assert model.compute_credit_score(0.0) == config.CREDIT_SCORE_MAX
    assert model.compute_credit_score(1.0) == config.CREDIT_SCORE_MIN


def test_compute_credit_score_midpoint():
    score = model.compute_credit_score(0.5)
    expected = int(round((config.CREDIT_SCORE_MAX + config.CREDIT_SCORE_MIN) / 2))
    assert score == expected


def test_decide_auto_approve_below_cutoff():
    decision, tier = model.decide(config.CUTOFF_APPROVE - 0.001)
    assert decision == "AUTO-APPROVE"
    assert "Low Risk" in tier


def test_decide_manual_review_between_cutoffs():
    midpoint = (config.CUTOFF_APPROVE + config.CUTOFF_REJECT) / 2
    decision, tier = model.decide(midpoint)
    assert decision == "MANUAL REVIEW"
    assert "Medium Risk" in tier


def test_decide_auto_reject_above_cutoff():
    decision, tier = model.decide(config.CUTOFF_REJECT + 0.05)
    assert decision == "AUTO-REJECT"
    assert "High Risk" in tier


def test_reason_codes_default_when_no_flags():
    codes = model.reason_codes({}, {})
    assert codes == ["No critical risk flags detected; standard portfolio profile"]


def test_reason_codes_flags_high_credit_income_ratio():
    codes = model.reason_codes({}, {"CREDIT_INCOME_RATIO": 5.0})
    assert "High Credit-to-Income Ratio (Excessive Leverage)" in codes


def test_reason_codes_capped_at_three():
    features = {
        "EXT_SOURCE_MEAN": 0.1,
        "INST_LATE_RATIO": 0.5,
        "CREDIT_INCOME_RATIO": 10.0,
        "CC_AVG_UTILIZATION": 0.9,
        "PREV_REFUSED_RATIO": 0.9,
        "EMPLOYED_YEARS": 0.1,
    }
    codes = model.reason_codes({}, features)
    assert len(codes) == 3
