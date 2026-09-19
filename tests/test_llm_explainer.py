"""
Unit tests for llm_explainer.py's provider-agnostic logic (language
detection, context/prompt building). These run with no network access and
no API key.

The one test that actually calls Gemini (`test_live_explain_result`) is
skipped automatically unless GEMINI_API_KEY is set, mirroring how
test_api.py skips /score until model artifacts exist.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_explainer import (  # noqa: E402
    LLMNotConfiguredError,
    build_combined_context,
    build_context,
    build_system_prompt,
    detect_language,
    explain_result,
)


def test_detect_language_arabic():
    assert detect_language("ليه الطلب اتحول للمراجعة اليدوية؟") == "ar"


def test_detect_language_english():
    assert detect_language("Why was this application flagged?") == "en"


def test_detect_language_empty_defaults_to_english():
    assert detect_language("") == "en"
    assert detect_language(None) == "en"


def test_build_context_labels_known_types():
    text = build_context("fraud", {"fraud_risk_level": "LOW"})
    assert "FRAUD DETECTION RESULT" in text
    assert "fraud_risk_level" in text


def test_build_context_unknown_type_falls_back():
    text = build_context("something_else", {"a": 1})
    assert "### DATA" in text


def test_build_combined_context_skips_empty_blocks():
    blocks = [
        {"type": "fraud", "data": {"fraud_risk_level": "HIGH"}},
        {"type": "credit_risk", "data": {}},  # empty -- should be skipped
        {"type": "portfolio", "data": None},  # missing -- should be skipped
    ]
    text = build_combined_context(blocks)
    assert "FRAUD DETECTION RESULT" in text
    assert "PORTFOLIO" not in text
    assert "CREDIT RISK" not in text


def test_build_combined_context_all_empty():
    text = build_combined_context([])
    assert "no result data was supplied" in text


def test_build_system_prompt_language_switch():
    ar_prompt = build_system_prompt("ar")
    en_prompt = build_system_prompt("en")
    assert "Arabic" in ar_prompt
    assert "English" in en_prompt
    assert ar_prompt != en_prompt


def test_explain_result_without_api_key_raises_configured_error(monkeypatch):
    monkeypatch.setattr("llm_explainer.GEMINI_API_KEY", "")
    with pytest.raises(LLMNotConfiguredError):
        explain_result(
            blocks=[{"type": "fraud", "data": {"fraud_risk_level": "LOW"}}],
            question="why is this low risk?",
        )


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="requires a live GEMINI_API_KEY"
)
def test_live_explain_result():
    result = explain_result(
        blocks=[
            {
                "type": "fraud",
                "data": {
                    "fraud_risk_level": "LOW",
                    "fraud_risk_score": 0.05,
                    "explainable_ai": {
                        "executive_summary_en": "Application passed all checks."
                    },
                },
            }
        ],
        question="Why is this applicant low risk?",
        lang="en",
    )
    assert result["language"] == "en"
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
