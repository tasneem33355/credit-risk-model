"""
Unified Fast API Service for Credit Risk & Application Fraud Analysis
======================================================================
Platform: Smart Financing & Credit Request Analysis Platform (ZAWOLF)
"""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

import os
import sys

# Ensure local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs
from llm_explainer import explain_result, LLMNotConfiguredError, LLMRequestError

app = FastAPI(
    title="ZAWOLF Credit Decisioning & Fraud Prevention API",
    description="End-to-End Enterprise API evaluating Application Document Fraud and Credit Risk Scoring.",
    version="2.0.0"
)

fraud_engine = CreditFraudEngine()


class ExplainBlock(BaseModel):
    """One piece of result data to ground the explanation in."""

    type: str = Field(
        ...,
        description="One of: credit_risk, fraud, portfolio, custom",
    )
    data: Dict[str, Any]


class ExplainRequest(BaseModel):
    """Input to POST /explain.

    Pass whichever CrediX result(s) you have available -- the fraud
    engine's `evaluate()` output, the PD model's `/score` response, a
    portfolio-analytics KPI dict, or any other JSON you want explained.
    Leave `question` empty to get a one-shot plain-language summary
    instead of answering a specific question.
    """

    blocks: List[ExplainBlock] = Field(
        ..., description="Result data to explain, e.g. fraud_engine.evaluate() output"
    )
    question: Optional[str] = Field(
        default=None,
        description="A free-form follow-up question. Omit for an auto-generated summary.",
    )
    lang: Optional[str] = Field(
        default=None,
        description="'ar' or 'en'. Auto-detected from `question` if omitted.",
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description='Prior chat turns as [{"role": "user"|"assistant", "content": str}, ...]',
    )


class ExplainResponse(BaseModel):
    answer: str
    language: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ZAWOLF Unified Credit & Fraud Engine",
        "version": "2.0.0",
        "features_contract": "application_data_contract.schema.json (v2)"
    }


@app.post("/api/v1/fraud/evaluate")
def evaluate_fraud(payload: Dict[str, Any]):
    """
    Evaluates incoming Application JSON against all 4 Forensic & Consistency layers.
    Returns: Fraud score, CBE reason codes, verification checklist, and risk level.
    """
    try:
        assessment = fraud_engine.evaluate(payload)
        return assessment
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating application fraud: {str(e)}")


@app.post("/api/v1/application/enrich")
def enrich_application(payload: Dict[str, Any]):
    """
    Enriches the input JSON payload with fraud assessment and credibility haircut,
    returning a complete contract payload ready for downstream risk storage or inference.
    """
    try:
        enriched = fraud_engine.enrich_payload(payload)
        return enriched
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error enriching payload: {str(e)}")


@app.post("/api/v1/application/evaluate-end-to-end")
def evaluate_end_to_end(payload: Dict[str, Any]):
    """
    Complete End-to-End Pipeline:
      1. Runs Fraud & Forensic consistency checks
      2. If CRITICAL -> Rejects application immediately with reason codes
      3. If PASSED -> Transforms into 413-feature vector and prepares for Credit Risk Decision
    """
    assessment = fraud_engine.evaluate(payload)
    app_features, history_features = adapt_application_to_model_inputs(payload)

    is_critical_fraud = assessment["fraud_risk_level"] == "CRITICAL"
    
    response = {
        "application_id": payload.get("application_id"),
        "fraud_assessment": assessment,
        "credit_risk_evaluation": {
            "eligible_for_credit_scoring": not is_critical_fraud,
            "status": "HALTED_DUE_TO_SUSPECTED_FRAUD" if is_critical_fraud else "PROCEEDED_TO_RISK_SCORING",
            "model_ready_features_count": len(app_features) + len(history_features),
            "internal_history_missing": bool(history_features.get("INTERNAL_HISTORY_MISSING", 1.0)),
            "risk_adjusted_salary": assessment["downstream_risk_feeder"]["risk_adjusted_salary"]
        }
    }
    return response


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest):
    """
    AI explainability layer: turns fraud-engine / credit-risk / portfolio
    result JSON into a plain-language explanation, or answers a free-form
    follow-up question grounded strictly in the data you pass in `blocks`.

    Any system (dashboard, core-banking, a mobile app for underwriters)
    can call this with whatever result it just received from
    /api/v1/fraud/evaluate, /score, or a portfolio-analytics snapshot.
    """
    try:
        result = explain_result(
            blocks=[b.dict() for b in payload.blocks],
            question=payload.question,
            lang=payload.lang,
            history=payload.history,
        )
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
