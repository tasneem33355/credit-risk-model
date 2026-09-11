from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    """Input to POST /score.

    `application` holds the raw application-level fields, using the same
    column names as Home Credit's application_train.csv (e.g. AMT_CREDIT,
    AMT_INCOME_TOTAL, AMT_ANNUITY, EXT_SOURCE_1/2/3, DAYS_BIRTH,
    DAYS_EMPLOYED, NAME_CONTRACT_TYPE, ...). Any field the model expects but
    that is missing here is imputed with the training-set median (numeric)
    or "Missing" (categorical) -- the same rule used in training.

    `history_features` is optional and lets a caller pass pre-computed
    aggregate features (bureau history, previous-loan history, POS-CASH,
    installments, credit-card) when those are available from a feature
    store. Any such feature NOT supplied here defaults to 0, which is the
    correct treatment for a genuinely new-to-bank applicant with no
    history, but understates risk for an applicant whose history simply
    wasn't joined in. See docs/model_card.md ("Serving-time limitations").
    """

    application: Dict[str, Any] = Field(..., description="Raw application-level fields")
    history_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional pre-computed bureau / previous-loan / installments / POS / credit-card aggregates",
    )


class CreditDecision(BaseModel):
    credit_score: int
    probability_of_default: str
    decision: str
    risk_tier: str
    reason_codes: List[str]
    model_version: str
