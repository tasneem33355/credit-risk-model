import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config

client = TestClient(app)

ARTIFACTS_PRESENT = os.path.exists(os.path.join(config.MODEL_DIR, "xgb_final.pkl"))


def test_health_endpoint_responds():
    response = client.get("/health")
    # 200 once model/train.py has been run, 503 if artifacts are missing --
    # both are valid, well-formed responses for this endpoint.
    assert response.status_code in (200, 503)


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="Run model/train.py first to generate model artifacts")
def test_score_endpoint_returns_a_decision():
    payload = {
        "application": {
            "AMT_CREDIT": 500000,
            "AMT_INCOME_TOTAL": 150000,
            "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 450000,
            "EXT_SOURCE_1": 0.7,
            "EXT_SOURCE_2": 0.65,
            "EXT_SOURCE_3": 0.72,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -2000,
            "CNT_FAM_MEMBERS": 3,
            "CNT_CHILDREN": 1,
            "NAME_CONTRACT_TYPE": "Cash loans",
        }
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"AUTO-APPROVE", "MANUAL REVIEW", "AUTO-REJECT"}
    assert 300 <= body["credit_score"] <= 850
    assert len(body["reason_codes"]) >= 1
