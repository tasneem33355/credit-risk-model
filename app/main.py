from fastapi import FastAPI, Header, HTTPException

from . import config, model
from .schemas import CreditApplication, CreditDecision

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Production scoring endpoint for the XGBoost + LightGBM credit-default model.",
    version="1.0.0",
)


def _check_api_key(x_api_key: str | None) -> None:
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    try:
        bundle = model.load_models()
        return {
            "status": "ok",
            "model_version": model.MODEL_VERSION,
            "n_features": len(bundle.feature_names),
            "metrics": bundle.metrics,
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run model/train.py to generate them first.",
        )


@app.post("/score", response_model=CreditDecision)
def score(payload: CreditApplication, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    try:
        result = model.score_application(payload.application, payload.history_features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result.pop("used_history_defaults", None)
    return result
