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

# Basel-style P&L assumptions used only for the offline portfolio simulation,
# not for scoring individual applicants.
LOSS_GIVEN_DEFAULT = 0.45
NET_INTEREST_MARGIN = 0.10

CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850
