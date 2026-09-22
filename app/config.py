import os

from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "model/artifacts")
API_KEY = os.getenv("API_KEY", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ----------------------------------------------------------------------
# Bank policy cutoffs on predicted Probability of Default (PD).
# Derived from the portfolio / P&L simulation in docs/model_card.md.
# Update these when the bank's risk appetite or NPL target changes --
# they are a business decision, not a statistical constant.
# ----------------------------------------------------------------------
CUTOFF_APPROVE = 0.0723   # PD below this -> AUTO-APPROVE   (portfolio NPL ~ 3.14%)
CUTOFF_REJECT = 0.20      # PD at/above this -> AUTO-REJECT (high risk)
# Between the two cutoffs -> MANUAL REVIEW (collateral / guarantor / reduced limit)

# Ensemble blend weights (XGBoost / LightGBM), selected by validation ROC-AUC search
DEFAULT_W_XGB = 0.3
DEFAULT_W_LGB = 0.7

# Basel-style P&L assumptions from docs/model_card.md §4. Used both for the
# offline portfolio simulation and for the per-applicant business-impact
# figures returned by app/model.py: compute_business_impact().
LOSS_GIVEN_DEFAULT = 0.45
NET_INTEREST_MARGIN = 0.10

# Reference points from docs/model_card.md §3 (acceptance rate -> portfolio
# NPL rate), used to give each decision a portfolio-level frame of reference.
# Not recomputed per request -- these are the historical validation-set
# simulation results and should be refreshed if model/train.py is rerun.
MARKET_BAD_RATE = 0.0807
PORTFOLIO_NPL_BY_ACCEPTANCE = {0.40: 0.0194, 0.50: 0.0235, 0.60: 0.0281, 0.70: 0.0343, 0.80: 0.0430, 0.85: 0.0485}

CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850
