"""
CrediX / ZAWOLF Unified Enterprise API Gateway
==============================================
Single Production Gateway combining:
  1. 5-Layer Forensic Fraud Detection Engine
  2. XGBoost + LightGBM Credit Default Risk Scoring
  3. End-to-End Automated Underwriting Pipeline
  4. Generative AI Explainability (LLM Assistant)
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fraud_engine import CreditFraudEngine
from adapter import adapt_application_to_model_inputs
from app import model, config
from app.schemas import CreditApplication, CreditDecision

# Optional LLM Explainer Import
try:
    from llm_explainer import explain_result, LLMNotConfiguredError, LLMRequestError
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

app = FastAPI(
    title="CrediX Enterprise Credit & Fraud Decisioning API",
    description="Unified Production API for Forensic Fraud Detection, Credit Risk Scoring, and AI Explainability.",
    version="3.0.0"
)

# -----------------------------------------------------------------------------
# 1. CORS Middleware (Enables Next.js / Vercel Frontend Integration)
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows localhost:3000, Vercel deployments, and production domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fraud_engine = CreditFraudEngine()


def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Validates API Key if configured in environment."""
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# -----------------------------------------------------------------------------
# 2. Schemas for LLM Explainer
# -----------------------------------------------------------------------------
class ExplainBlock(BaseModel):
    type: str = Field(..., description="One of: credit_risk, fraud, portfolio, custom")
    data: Dict[str, Any]


class ExplainRequest(BaseModel):
    blocks: List[ExplainBlock] = Field(..., description="Result data to explain")
    question: Optional[str] = Field(default=None, description="Follow-up question or omit for auto summary")
    lang: Optional[str] = Field(default=None, description="'ar' or 'en'")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Prior chat history")


class ExplainResponse(BaseModel):
    answer: str
    language: str


# -----------------------------------------------------------------------------
# 3. Health & Status Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "CrediX Enterprise Underwriting Gateway",
        "status": "online",
        "documentation": "/docs",
        "version": "3.0.0"
    }


@app.get("/health")
def health():
    credit_models_ready = False
    try:
        bundle = model.load_models()
        credit_models_ready = bool(bundle.xgb_model and bundle.lgb_model)
    except Exception:
        credit_models_ready = False

    return {
        "status": "healthy",
        "service": "CrediX Unified Decisioning Gateway",
        "fraud_engine_status": "online",
        "credit_risk_model_ready": credit_models_ready,
        "llm_assistant_available": LLM_AVAILABLE,
        "version": "3.0.0"
    }


# -----------------------------------------------------------------------------
# 4. Standalone Credit Risk Scoring Endpoint
# -----------------------------------------------------------------------------
@app.post("/score", response_model=CreditDecision)
def score_credit(payload: CreditApplication, x_api_key: Optional[str] = Header(default=None)):
    """
    Direct Credit Risk Scoring Endpoint (Home Credit XGBoost + LightGBM Blend).
    Calculates PD, Credit Score (300-850), Risk Tier, and Decision Cutoffs.
    """
    _check_api_key(x_api_key)
    try:
        result = model.score_application(payload.application, payload.history_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Credit scoring error: {str(exc)}")
    
    result.pop("used_history_defaults", None)
    return result


# -----------------------------------------------------------------------------
# 5. Standalone Forensic Fraud Evaluation Endpoint
# -----------------------------------------------------------------------------
@app.post("/api/v1/fraud/evaluate")
def evaluate_fraud(payload: Dict[str, Any]):
    """
    Evaluates application JSON against the 5-Layer Forensic Fraud Architecture.
    """
    try:
        assessment = fraud_engine.evaluate(payload)
        return assessment
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating fraud: {str(e)}")


@app.post("/api/v1/application/enrich")
def enrich_application(payload: Dict[str, Any]):
    """Enriches raw payload with fraud metadata and credibility haircut."""
    try:
        return fraud_engine.enrich_payload(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error enriching payload: {str(e)}")


# -----------------------------------------------------------------------------
# 6. Complete End-to-End Enterprise Underwriting Pipeline (Primary Gateway)
# -----------------------------------------------------------------------------
@app.post("/api/v1/application/evaluate-end-to-end")
def evaluate_end_to_end(payload: Dict[str, Any]):
    """
    The Crown-Jewel Production Underwriting Endpoint:
      Step 1: Executes 5-Layer Forensic Fraud Gating
      Step 2: If Fatal Fraud (CRITICAL) -> Halts, rejects, and emits CBE audit reason codes
      Step 3: If Genuine / Recoverable (LOW / MEDIUM) -> Applies Credibility Haircut,
              adapts 413 features, and executes Credit Risk Scoring (XGBoost + LightGBM).
    """
    try:
        # 1. Run 5-Layer Fraud Analysis
        assessment = fraud_engine.evaluate(payload)
        is_critical_fraud = assessment.get("fraud_risk_level") == "CRITICAL"
        
        # 2. Extract adapted features for credit model
        app_features, history_features = adapt_application_to_model_inputs(payload)

        credit_decision = None
        underwriting_status = "HALTED_DUE_TO_SUSPECTED_FRAUD" if is_critical_fraud else "PROCEEDED_TO_RISK_SCORING"

        # 3. If passed fraud gating, execute Credit Risk Scoring
        if not is_critical_fraud:
            try:
                credit_decision = model.score_application(app_features, history_features)
                credit_decision.pop("used_history_defaults", None)
            except Exception as e:
                credit_decision = {
                    "error": f"Credit scoring model error: {str(e)}",
                    "note": "Verify that credit risk model artifacts exist in model/artifacts/"
                }

        # 4. Formulate Unified Response for Frontend
        return {
            "application_id": payload.get("application_id", "N/A"),
            "underwriting_status": underwriting_status,
            "final_action": "REJECT" if is_critical_fraud else credit_decision.get("decision", "EVALUATED") if credit_decision else "MANUAL_REVIEW",
            "fraud_layer": {
                "risk_level": assessment.get("fraud_risk_level"),
                "fraud_risk_score": assessment.get("fraud_risk_score"),
                "recommended_action": assessment.get("recommended_action"),
                "credibility_haircut_percentage": assessment.get("downstream_risk_feeder", {}).get("haircut_percentage", 0.0),
                "risk_adjusted_salary": assessment.get("downstream_risk_feeder", {}).get("risk_adjusted_salary"),
                "total_rule_violations": assessment.get("metrics", {}).get("total_violations_count", 0),
                "critical_violations": assessment.get("metrics", {}).get("critical_violations_count", 0),
                "full_assessment": assessment
            },
            "credit_layer": credit_decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"End-to-End processing error: {str(e)}")


# -----------------------------------------------------------------------------
# 7. AI Explainability Layer (LLM Assistant)
# -----------------------------------------------------------------------------
@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest):
    """Answers underwriter questions and translates complex decisions into plain language."""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM explainer module not configured")
    try:
        result = explain_result(
            blocks=[b.dict() for b in payload.blocks],
            question=payload.question,
            lang=payload.lang,
            history=payload.history,
        )
        return result
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
