# Model Card — Credit Default Risk Scoring Model

## 1. Overview

| | |
|---|---|
| **Task** | Binary classification — probability that an applicant defaults on a loan (TARGET = 1) |
| **Training data** | Home Credit Default Risk (Kaggle) — `application_train.csv` joined with `previous_application`, `installments_payments`, `POS_CASH_balance`, `bureau`, `bureau_balance`, `credit_card_balance` |
| **Rows** | 307,511 applications, 8.07% historical default rate |
| **Features** | 413, after one-hot encoding of application-level categoricals |
| **Algorithm** | Weighted blend of XGBoost (w=0.3) and LightGBM (w=0.7), each independently early-stopped on a held-out validation set |
| **Validation split** | 80/20 stratified, random_state=42 |
| **Model version** | `credit-risk-xgb-lgb-blend-v1` |

## 2. Performance

| Metric | Value | Benchmark |
|---|---|---|
| ROC-AUC | 0.7895 | — |
| Gini coefficient (`2×AUC−1`) | 57.90% | >50% = Strong Rating Model (central-bank / Basel II-III convention) |
| K-S statistic | 44.19% | >40% = high separation power |
| PR-AUC | reported in `model/artifacts/metrics.json` after training | — |
| Baseline (market) bad rate | 8.07% | — |

For context: the #1 finishing entry among 7,000+ teams in the original Kaggle
competition this dataset is drawn from scored ROC-AUC = 0.805. A model in this
range is expected to carry a meaningful number of false positives and false
negatives at any single fixed threshold — that is a property of the 8%
class-imbalance and the inherent noise in individual repayment behavior, not
evidence of a broken model. See §4 for why bank scoring models are not
evaluated against a single confusion matrix.

## 3. Portfolio simulation

Applicants are ranked by predicted PD (lowest risk first) and accepted up to
a chosen acceptance rate, rather than compared against a single 0.5
threshold:

| Acceptance Rate | Portfolio NPL Rate | Market NPL Rate | NPL Reduction | Bad Loans Prevented (of 61,503 test applicants) |
|---:|---:|---:|---:|---:|
| 40% | 1.94% | 8.07% | 76.0% | 4,488 |
| 50% | 2.35% | 8.07% | 70.8% | 4,241 |
| 60% | 2.81% | 8.07% | 65.2% | 3,928 |
| 70% | 3.43% | 8.07% | 57.6% | 3,490 |
| 80% | 4.30% | 8.07% | 46.7% | 2,849 |
| 85% | 4.85% | 8.07% | 40.0% | 2,431 |

## 4. P&L simulation (Basel-style assumptions)

Assumptions: Net Interest Margin = 10% of `AMT_CREDIT` on performing loans;
Loss Given Default (LGD) = 45% of `AMT_CREDIT` on defaulted loans.

| Acceptance | NPL Rate | Net Profit | Return on Portfolio |
|---:|---:|---:|---:|
| 70% | 3.43% | $2.17B | 8.12% |
| 85% (profit-maximizing) | 4.85% | $2.36B | 7.40% |
| 100% (no model) | 8.07% | $2.15B | 5.85% |

Using the model at a 70% acceptance rate reduces default losses from $1.25B
(no model) to $410M — a $838M reduction — while the profit-maximizing 85%
acceptance strategy adds +$214M (+10.0%) of net profit versus not scoring
applicants at all. These are simulation outputs on the historical validation
set, not a guarantee of future portfolio performance.

## 5. Decision policy

| PD range | Decision | Treatment |
|---|---|---|
| PD < 7.23% | AUTO-APPROVE | Immediate approval |
| 7.23% ≤ PD < 20% | MANUAL REVIEW | Request collateral / guarantor / reduced limit |
| PD ≥ 20% | AUTO-REJECT | Decline, with adverse-action reason codes |

These cutoffs (`app/config.py: CUTOFF_APPROVE`, `CUTOFF_REJECT`) are a
business/risk-appetite choice derived from §3–4, not a statistical output —
revisit them whenever the bank's target NPL or acceptance volume changes.

## 6. Adverse action reason codes

Any AUTO-REJECT or MANUAL REVIEW decision returns up to 3 rule-based reason
codes (`app/model.py: reason_codes()`), as required under fair-lending /
adverse-action disclosure regulation:

- Low External Bureau Score / Credit History Rating
- Historical Late Payment Record on Past Loans
- High Credit-to-Income Ratio (Excessive Leverage)
- High Revolving Credit Card Balance Utilization
- High Previous Loan Rejection History
- Short Employment Tenure (< 1 Year)

## 7. Serving-time limitations (read before production use)

- **History features at scoring time.** The model was trained with bureau,
  previous-application, installments, POS-CASH, and credit-card aggregate
  features. `POST /score` computes the application-level block exactly, but
  defaults history-derived features to 0 unless the caller supplies them via
  `history_features`. In production this block should be populated from a
  feature store / data warehouse join keyed on the applicant, not left at
  the default — leaving it at 0 understates risk for any applicant who has
  real history.
- **Population shift.** The training data is a single geography/time window
  (Home Credit's historical portfolio). Before scoring a materially
  different population (e.g. a different country's applicants), validate
  Gini/K-S on a local labeled sample and fine-tune if they degrade — see the
  retraining note below.
- **Monitoring.** Re-validate Gini/K-S/PSI on a rolling window (quarterly is
  a common cadence for credit models) and retrain if population or macro
  conditions shift materially.
- **Regulatory approval.** Any bank deploying this model against real credit
  decisions is expected to submit it — including this model card, the
  portfolio simulation, and reason-code logic — for internal model-risk
  validation and, where applicable, central-bank approval before production
  use.

## 8. Retraining / fine-tuning on new data

`model/train.py` reproduces this pipeline end-to-end against any directory
of Home-Credit-formatted CSVs. For a new institution's data:

1. **Small sample (< ~5,000 records):** continue training the existing
   boosters (`init_model=`) rather than fitting from scratch, so the model
   keeps what it learned from the larger dataset.
2. **Medium sample (~5,000–50,000 records):** fit a new lightweight model
   on the local data, using this model's predictions as an additional input
   feature (stacking) — typically more robust than pure fine-tuning at this
   size.
3. Map local fields to this model's feature names first (e.g. a local
   bureau score standing in for `EXT_SOURCE_*`) — see `model/features.py`
   for the exact feature definitions to replicate.

## 9. Files

- `notebooks/credit_risk_model.ipynb` — original development notebook
- `model/features.py` — feature engineering (shared by training and serving)
- `model/train.py` — reproducible training entry point
- `app/` — FastAPI scoring service
- `model/artifacts/metrics.json` — exact metrics from the most recent training run
