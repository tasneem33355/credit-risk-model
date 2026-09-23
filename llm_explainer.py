"""
CrediX AI Explainability Layer
===============================
A thin, provider-agnostic layer that turns the structured JSON output of
CrediX's ML components -- the credit-risk PD scorer (app/model.py), the
5-layer fraud engine (fraud_engine.py), and the portfolio / risk-lab
analytics (portfolio_analytics.py, dashboard.py) -- into plain-language,
business-friendly explanations, and answers free-form follow-up questions
grounded strictly in that data.

Used by two front doors:
  - dashboard.py            -> "AI Assistant" chat tab (interactive, Streamlit)
  - api.py (/explain)       -> a stateless HTTP endpoint any system can call

Provider: Google Gemini, because it currently has the most usable free
tier (no credit card, generous daily quota) of the major hosted LLM APIs --
see docs/llm_layer.md for how to get a key. The only function that talks
to the network is `call_llm()`; swapping providers later (OpenAI, Claude,
a local model, ...) means rewriting that one function and leaving prompt
building, context formatting, and chat history untouched.

This module deliberately has ONE non-stdlib dependency (`requests`,
already used elsewhere in this repo) so it can be imported from both the
FastAPI services and the Streamlit dashboard without pulling in a heavy
SDK.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

try:  # pragma: no cover - optional convenience, mirrors app/config.py
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("credix.llm")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Configuration (all overridable via environment / .env -- see .env.example)
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# gemini-3.1-flash-lite is Google's current free-tier workhorse model
# (stable since May 2026). If Google renames/retires it, update GEMINI_MODEL
# in .env -- nothing else in this file needs to change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "900"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Retry policy for transient upstream failures (rate limiting / momentary
# outages) -- NOT for bad requests (4xx other than 429) or bad API keys,
# which are retried at 0 extra attempts since retrying won't help.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY_SECONDS = float(os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "1.5"))
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMNotConfiguredError(RuntimeError):
    """Raised when no usable API key/provider is configured."""


class LLMRequestError(RuntimeError):
    """Raised when the upstream LLM API call fails or returns something
    this module doesn't know how to parse."""


# ---------------------------------------------------------------------------
# Language detection -- tiny, dependency-free heuristic.
# Good enough to route "reply in Arabic vs. English"; not a general-purpose
# language identifier.
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: Optional[str]) -> str:
    """Returns 'ar' if `text` contains Arabic script, else 'en'."""
    if text and _ARABIC_RE.search(text):
        return "ar"
    return "en"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """\
You are "Tia Explain", the built-in AI explainability assistant inside \
CrediX -- an Egyptian bank's embedded credit-decisioning platform that \
combines a Probability-of-Default (PD) credit-risk model (XGBoost + \
LightGBM ensemble), a 5-layer application fraud-detection engine, and a \
portfolio / risk-lab analytics module (ECL, VaR, vintage curves, PSI \
drift, pricing).

Your audience is bank staff -- underwriters, credit officers, risk and \
portfolio managers, branch managers -- who trust the numbers but are NOT \
necessarily data scientists. Your job is to explain what the model \
outputs mean for a real credit decision, not to lecture on machine \
learning theory.

Ground rules, in order of importance:
1. GROUNDING: Base every claim strictly on the JSON DATA block(s) given to \
you below this prompt. Never invent a number, metric, or fact that is not \
present in that data. If the user asks something the provided data cannot \
answer, say so plainly and name what additional data would be needed -- \
never guess or fabricate to sound complete.
2. PLAIN LANGUAGE, PROFESSIONAL TONE: Write like a sharp risk analyst \
briefing a business colleague, not like a textbook. Prefer everyday words \
over statistical jargon (e.g. say "the model found a big jump in this \
customer's payment history that doesn't match how they normally behave" \
rather than dwelling on Isolation Forest internals) -- but stay precise \
and use the exact figures from the data (percentages, scores, EGP \
amounts, thresholds) rather than vague qualifiers alone.
3. STRUCTURE: Keep answers focused -- a short lead sentence with the \
bottom line, then a few short bullet points or sentences with the \
supporting reasons/numbers. Avoid long unbroken paragraphs and avoid \
restating the entire JSON payload back at the user.
4. HONESTY ABOUT ROLE: You explain and interpret; you do not override the \
platform's decision engine. If relevant, you may note once that this is \
decision support and a human underwriter/committee holds final sign-off \
-- do not repeat that disclaimer in every message.
5. NO SPECULATIVE ADVICE: Do not invent regulatory, legal, or credit-policy \
advice beyond what the data and CrediX's own reason codes/policy fields \
already state.
6. SCOPE: Politely decline (in one short sentence) topics that have \
nothing to do with the CrediX result data you were given, and redirect \
the user back to what you can help with.
7. CREDIT RISK RESULTS -- LEAD WITH BUSINESS IMPACT: When a CREDIT RISK \
RESULT block is present, do not stop at the PD percentage and decision \
label. If the data includes a `business_impact` field, translate it into \
what a credit committee actually asks: the expected loss if this loan \
defaults, the expected profit if it performs, and the resulting \
risk-adjusted return -- in EGP, not just percentages. If a \
`portfolio_context` field is present, use it to frame the individual \
decision against the bank's historical portfolio performance (NPL rate, \
market baseline) instead of presenting the PD as an isolated number. This \
is how the bank itself reasons about a credit decision (see the model \
card's portfolio and P&L simulation sections) -- mirror that framing \
rather than defaulting to model-metric language (ROC-AUC, feature \
importance, etc.) unless the user explicitly asks for the technical view.
"""


def build_system_prompt(lang: str) -> str:
    lang_line = (
        "Respond in professional, business-appropriate Egyptian-friendly "
        "Modern Standard Arabic (اللغة العربية الفصحى المبسطة، بأسلوب مهني وواضح)."
        if lang == "ar"
        else "Respond in clear, professional business English."
    )
    return BASE_SYSTEM_PROMPT + "\n" + lang_line


# ---------------------------------------------------------------------------
# Context builders -- turn raw model/engine output into compact text blocks
# ---------------------------------------------------------------------------

_BLOCK_LABELS = {
    "credit_risk": "CREDIT RISK RESULT (Probability-of-Default scoring model: "
    "XGBoost + LightGBM ensemble)",
    "fraud": "FRAUD DETECTION RESULT (CrediX 5-layer application fraud & "
    "consistency engine)",
    "portfolio": "PORTFOLIO / RISK-LAB ANALYTICS SNAPSHOT (static core-banking "
    "sample workbook -- the bank's existing book)",
    "scored_portfolio": "LIVE SCORED-APPLICATION PORTFOLIO (aggregated across every "
    "application this deployment's trained PD model has actually scored so far -- "
    "distinct from the static core-banking snapshot above)",
    "custom": "ADDITIONAL CONTEXT DATA",
}


def _truncate_json(data: Any, max_chars: int = 9000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated -- older/less relevant fields cut for length)"
    return text


def build_context(context_type: str, data: Dict[str, Any]) -> str:
    """
    context_type: one of "credit_risk", "fraud", "portfolio", "custom".
    data: the raw dict for that result, e.g. app/model.py's
    `score_application()` output, fraud_engine.py's `evaluate()` output, or
    a portfolio KPIs dict from portfolio_analytics.py.
    """
    label = _BLOCK_LABELS.get(context_type, "DATA")
    return f"### {label}\n```json\n{_truncate_json(data)}\n```"


def build_combined_context(blocks: List[Dict[str, Any]]) -> str:
    """blocks: list of {"type": ..., "data": {...}}. Empty/missing data
    blocks are skipped so callers can pass "whatever is available right
    now" without extra filtering."""
    parts = [build_context(b.get("type", "custom"), b["data"]) for b in blocks if b.get("data")]
    if not parts:
        return "### DATA\n(no result data was supplied for this request)"
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call -- Gemini (free tier). This is the ONLY function that talks to
# the network; swap providers by rewriting this one function.
# ---------------------------------------------------------------------------


def _messages_to_gemini_contents(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    contents = []
    for m in history:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})
    return contents


def call_llm(
    system_prompt: str,
    history: List[Dict[str, str]],
    temperature: float = TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    if LLM_PROVIDER != "gemini":
        raise LLMNotConfiguredError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Only 'gemini' is wired up today "
            "-- see docs/llm_layer.md for how to add another provider."
        )
    if not GEMINI_API_KEY:
        raise LLMNotConfiguredError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and put it in your .env file "
            "(see .env.example)."
        )

    url = GEMINI_ENDPOINT.format(model=GEMINI_MODEL, key=GEMINI_API_KEY)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": _messages_to_gemini_contents(history),
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    last_exc: Optional[Exception] = None

    for attempt in range(1, LLM_MAX_RETRIES + 2):  # e.g. 3 retries -> 4 total attempts
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Gemini request attempt %d failed (network error): %s", attempt, exc)
        else:
            if resp.status_code == 200:
                data = resp.json()
                try:
                    candidate = data["candidates"][0]
                    parts = candidate["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if not text:
                        raise KeyError("empty text")
                    if attempt > 1:
                        logger.info("Gemini request succeeded on attempt %d", attempt)
                    return text
                except (KeyError, IndexError) as exc:
                    finish_reason = (data.get("candidates") or [{}])[0].get("finishReason", "UNKNOWN")
                    raise LLMRequestError(
                        f"Unexpected/empty Gemini response (finish_reason={finish_reason}). "
                        f"Raw response: {json.dumps(data, ensure_ascii=False)[:800]}"
                    ) from exc

            if resp.status_code not in _RETRYABLE_STATUS_CODES:
                # Not worth retrying: bad request, invalid key, etc.
                raise LLMRequestError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

            last_exc = LLMRequestError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")
            logger.warning(
                "Gemini request attempt %d failed (status %d, retryable): %s",
                attempt, resp.status_code, resp.text[:200],
            )

        if attempt <= LLM_MAX_RETRIES:
            delay = LLM_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay)

    logger.error("Gemini request failed after %d attempts", LLM_MAX_RETRIES + 1)
    if isinstance(last_exc, requests.RequestException):
        raise LLMRequestError(f"Could not reach Gemini API after {LLM_MAX_RETRIES + 1} attempts: {last_exc}") from last_exc
    raise last_exc or LLMRequestError("Gemini API call failed for an unknown reason.")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def explain_result(
    blocks: List[Dict[str, Any]],
    question: Optional[str] = None,
    lang: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Main entry point used by both the dashboard chat tab and the /explain
    API endpoint.

    blocks:   [{"type": "credit_risk" | "fraud" | "portfolio" | "custom",
                "data": {...}}, ...]
              Pass whichever result(s) are available/relevant; unavailable
              ones can simply be left out.
    question: A free-form follow-up question. If None, produces a one-shot
              plain-language explanation of the supplied result(s) instead
              of answering a specific question.
    lang:     Force "ar" or "en". If None, it's inferred from `question`
              (falls back to "ar" for auto-summaries, since CrediX's
              primary users are Egyptian bank staff).
    history:  Prior turns as [{"role": "user"|"assistant", "content": str}],
              for a running chat. The new question/summary request is
              appended automatically -- callers don't need to do that.

    Returns {"answer": str, "language": "ar"|"en"}.
    Raises LLMNotConfiguredError / LLMRequestError on failure -- callers
    (dashboard.py, api.py) are expected to catch these and show a friendly
    message rather than letting the whole page/request crash.
    """
    if lang is None:
        lang = detect_language(question) if question else "ar"

    system_prompt = build_system_prompt(lang) + "\n\n" + build_combined_context(blocks)

    convo = list(history or [])
    if question:
        convo.append({"role": "user", "content": question})
    else:
        auto_prompt = (
            "من فضلك اشرح نتيجة التقييم دي للمستخدم بشكل واضح ومهني ومفهوم لغير "
            "المتخصصين في تحليل البيانات، مع التركيز على معنى الأرقام وتأثيرها على القرار."
            if lang == "ar"
            else "Please explain this result to the user in clear, professional, "
            "non-technical language, focusing on what the numbers mean for the decision."
        )
        convo.append({"role": "user", "content": auto_prompt})

    answer = call_llm(system_prompt, convo)
    return {"answer": answer, "language": lang}
