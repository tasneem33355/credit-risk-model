# CrediX Synthetic Data Patch & Generation Guide

## 1. Required Adjustments Summary
1. **Cold-Start Gap**: Add 35% New-to-Bank applicants with \internal_history_missing = 1\.
2. **Egyptian I-Score**: Add credit bureau fields (score 300-850, active loans, debt, max DPD).
3. **Application & Document Fraud**: Add cross-document checks (\income_mismatch_ratio\, employer match, National ID match).
4. **Target Calibration**: Include I-Score, DTI, and internal history penalty in default simulation.

---

## PROMPT 1: New Synthetic Dataset Generation (From Scratch)

\\	ext
# TASK: Generate Synthetic Dataset for Egyptian Embedded Credit Platform

1. Architecture: Embedded credit engine in an Egyptian commercial bank.
2. Customer Split:
   - 65% Returning Customers: is_returning_customer=True, internal_history_missing=0, full bank history (Accounts, Loans, Payments).
   - 35% New-to-Bank: is_returning_customer=False, internal_history_missing=1, internal history=0/empty.
3. Egyptian Features:
   - National ID: Age 21-65, Egyptian governorates, 14-digit ID.
   - Salary Slip: Net salary EGP 4k-80k, tenure, sector (Gov, Public, Private, MNC).
   - 6M Bank Statement: Inflow, avg balance, volatility, bounced cheques.
   - Egyptian I-Score: Score 300-850 (mean 670, std 75), total debt, active loans, max DPD.
   - Feature Drops: SOCIAL_CIRCLE_* dropped/0; EXT_SOURCE proxied via normalized I-Score.
4. Fraud Consistency Checks (Inject 5-8% anomalies):
   - income_mismatch_ratio = |salary - inflow| / salary (alert if > 0.50).
   - employer_name_match: Boolean (~3% False).
   - national_id_match_across_documents: Boolean (~1% False).
   - iscore_report_age_days: Integer (>30 is stale).
5. Ground Truth Target (defaulted_within_12m):
   Logit(P) = -2.2 - 0.006*(I_SCORE-600) + 2.0*(DTI-0.4) + 0.7*BUREAU_OVERDUE + 0.9*LATE_RATIO - 0.5*BALANCE_RATIO + 1.5*MISMATCH_FLAG + 0.35*HISTORY_MISSING + Noise(0, 0.35)
   TARGET = 1 if P > 0.18 else 0 (yields ~8-10% default rate).
6. Output: Tabular CSV or JSON array matching application_data_contract.schema.json (v2).
\
---

## PROMPT 2: CrediX Patch Request (To Update Existing CrediX)

\\	ext
# PATCH REQUEST: Update CrediX Synthetic Banking Dataset

Update CrediX generation logic with 4 required adjustments:
1. Cold-Start Segment in Credit_Applications (12,000 rows):
   - 65% Returning: is_returning_customer=True, internal_history_missing=0.
   - 35% New-to-Bank: is_returning_customer=False, internal_history_missing=1 (prior bank accounts/loans=0).
2. Add Egyptian I-Score to Credit_Applications:
   - iscore_credit_score (300-850, normal dist, mean 670, std 75).
   - iscore_score_tier (Excellent, Very Good, Good, Fair, Poor).
   - bureau_active_loans_count, bureau_total_debt, bureau_max_overdue_days, iscore_report_age_days.
3. Add Application Fraud Columns to Credit_Applications:
   - income_mismatch_ratio = abs(monthly_income - 6m_avg_inflow) / monthly_income (inject 5% > 0.50).
   - employer_name_match (Boolean, 3% False).
   - national_id_match_across_documents (Boolean, 1% False).
   - document_tampering_flag (Boolean, 2% True).
   - is_application_fraud (Target label for application/identity fraud).
4. Calibrate defaulted_within_12m:
   - Combine internal repayment history + external I-Score/DTI + risk penalty for internal_history_missing=1 and income_mismatch_ratio>0.25.
\