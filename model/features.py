"""
Feature engineering for the credit risk model.

This module mirrors the logic developed in notebooks/credit_risk_model.ipynb
so that training (model/train.py) and any future retraining run the exact
same transformations. Every function operates on the Home Credit Default
Risk tables (application_train.csv, previous_application.csv,
installments_payments.csv, POS_CASH_balance.csv, bureau.csv,
bureau_balance.csv, credit_card_balance.csv).

The pipeline builds four feature blocks and concatenates them into one
sparse matrix:
    1. Application-level features (core + secondary), one-hot encoded
       categoricals -> X_pos
    2. Bureau (credit bureau history) features -> X_bureau_final
    3. Credit card balance features -> X_cc_final
    4. (Previous application / installments / POS-CASH are merged into the
       application-level block before one-hot encoding, since they are
       purely numeric aggregates.)
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

# Columns selected empirically in the original notebook -- only these
# installments / POS-CASH aggregate columns made it into the winning
# feature set (X_pos).
INSTALL_SELECTED = [
    "INST_PAYMENT_COUNT", "INST_LATE_COUNT", "INST_SEVERE_LATE_COUNT", "INST_UNDERPAYMENT_COUNT",
    "INST_AVG_DAYS_LATE", "INST_MAX_DAYS_LATE", "INST_AVG_PAYMENT_RATIO", "INST_MIN_PAYMENT_RATIO",
    "INST_TOTAL_INSTALLMENT", "INST_TOTAL_PAYMENT", "INST_AVG_INSTALLMENT", "INST_AVG_PAYMENT",
    "INST_UNIQUE_PREV", "INST_LATE_RATIO", "INST_SEVERE_LATE_RATIO", "INST_UNDERPAYMENT_RATIO",
    "INST_TOTAL_PAYMENT_RATIO",
]

POS_SELECTED = [
    "POS_AVG_INSTALLMENT", "POS_AVG_INSTALLMENT_FUTURE", "POS_MIN_INSTALLMENT_FUTURE",
    "POS_MAX_INSTALLMENT_FUTURE", "POS_RECORD_COUNT", "POS_CONTRACT_COUNT",
    "POS_MONTH_COUNT", "POS_RECENT_1Y_RATIO", "POS_RECENT_2Y_RATIO",
]

BUREAU_MISSING_FLAG_COLS = [
    "BUREAU_AVG_DPD_RATIO", "BUREAU_MAX_STATUS",
    "BUREAU_AVG_MONTH_COUNT", "BUREAU_MAX_DAYS_CREDIT_ENDDATE",
    "BUREAU_DEBT_TO_CREDIT",
]
BUREAU_DROP_COLS = [
    "HAS_BUREAU_HISTORY", "HAS_BUREAU_BALANCE_DETAIL",
    "BUREAU_AVG_DPD_RATIO_MISSING", "BUREAU_AVG_MONTH_COUNT_MISSING",
    "BUREAU_MAX_DAYS_CREDIT_ENDDATE_MISSING",
]

CC_RATIO_MISSING_FLAG_COLS = ["CC_AVG_UTILIZATION", "CC_MAX_UTILIZATION", "CC_AVG_MIN_PAYMENT_RATIO"]
CC_DROP_COLS = [
    "CC_UNIQUE_PREV", "HAS_CC_HISTORY",
    "CC_AVG_UTILIZATION_MISSING", "CC_MAX_UTILIZATION_MISSING", "CC_AVG_MIN_PAYMENT_RATIO_MISSING",
]

EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


# ----------------------------------------------------------------------
# 1. Application-level features
# ----------------------------------------------------------------------
def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Age, employment, core financial ratios, missing-value indicators,
    and NaN/inf cleanup for the raw application table."""
    df = df.copy()

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    # DAYS_EMPLOYED == 365243 is a known Home Credit sentinel for "not employed"
    df["EMPLOYMENT_SENTINEL"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    days_employed_clean = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYED_YEARS"] = -days_employed_clean / 365.25

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)

    engineered_so_far = [
        "TARGET", "SK_ID_CURR", "AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYMENT_SENTINEL",
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "GOODS_CREDIT_RATIO",
    ]
    numeric_original = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in engineered_so_far
    ]
    for col in numeric_original:
        if df[col].isna().any():
            df[f"{col}_MISSING"] = df[col].isna().astype(np.int8)

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["TARGET", "SK_ID_CURR"]]
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna("Missing")

    return df


def add_secondary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Income/family burden ratios, EXT_SOURCE combinations, social-circle
    behavior. Requires add_core_features to have run first."""
    df = df.copy()

    df["INCOME_AFTER_ANNUITY"] = df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["CREDIT_PER_FAMILY_MEMBER"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["CHILDREN_FAMILY_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    df["EXT_SOURCE_MEAN"] = df[EXT_SOURCE_COLS].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[EXT_SOURCE_COLS].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[EXT_SOURCE_COLS].max(axis=1)

    df["SOCIAL_CIRCLE_OBS"] = df.get("OBS_30_CNT_SOCIAL_CIRCLE", 0) + df.get("OBS_60_CNT_SOCIAL_CIRCLE", 0)
    df["SOCIAL_CIRCLE_DEFAULTS"] = df.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) + df.get("DEF_60_CNT_SOCIAL_CIRCLE", 0)

    new_features = [
        "INCOME_AFTER_ANNUITY", "INCOME_PER_FAMILY_MEMBER", "CREDIT_PER_FAMILY_MEMBER",
        "CHILDREN_FAMILY_RATIO", "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX",
        "SOCIAL_CIRCLE_OBS", "SOCIAL_CIRCLE_DEFAULTS",
    ]
    df[new_features] = df[new_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


# ----------------------------------------------------------------------
# 2. Previous application aggregates
# ----------------------------------------------------------------------
def build_previous_application_features(path: str) -> pd.DataFrame:
    previous = pd.read_csv(path, encoding="latin-1")

    previous["PREV_APPROVED_FLAG"] = (previous["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    previous["PREV_REFUSED_FLAG"] = (previous["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    previous["PREV_CANCELED_FLAG"] = (previous["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.int8)
    previous["PREV_DAYS_TO_CURRENT"] = -previous["DAYS_DECISION"]
    previous["PREV_RECENT_1Y"] = (previous["PREV_DAYS_TO_CURRENT"] <= 365).astype(np.int8)
    previous["PREV_RECENT_2Y"] = (previous["PREV_DAYS_TO_CURRENT"] <= 730).astype(np.int8)
    previous["PREV_CREDIT_TO_APPLICATION"] = previous["AMT_CREDIT"] / previous["AMT_APPLICATION"].replace(0, np.nan)
    previous["PREV_ANNUITY_TO_CREDIT"] = previous["AMT_ANNUITY"] / previous["AMT_CREDIT"].replace(0, np.nan)

    prev_agg = previous.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("PREV_APPROVED_FLAG", "sum"),
        PREV_REFUSED_COUNT=("PREV_REFUSED_FLAG", "sum"),
        PREV_CANCELED_COUNT=("PREV_CANCELED_FLAG", "sum"),
        PREV_AVG_CREDIT_TO_APPLICATION=("PREV_CREDIT_TO_APPLICATION", "mean"),
        PREV_AVG_ANNUITY=("AMT_ANNUITY", "mean"),
        PREV_AVG_CREDIT=("AMT_CREDIT", "mean"),
        PREV_AVG_APPLICATION=("AMT_APPLICATION", "mean"),
        PREV_AVG_CNT_PAYMENT=("CNT_PAYMENT", "mean"),
        PREV_AVG_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "mean"),
        PREV_RECENT_1Y_COUNT=("PREV_RECENT_1Y", "sum"),
        PREV_RECENT_2Y_COUNT=("PREV_RECENT_2Y", "sum"),
    )

    prev_agg["PREV_APPROVED_RATIO"] = prev_agg["PREV_APPROVED_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_REFUSED_RATIO"] = prev_agg["PREV_REFUSED_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_CANCELED_RATIO"] = prev_agg["PREV_CANCELED_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_RECENT_1Y_RATIO"] = prev_agg["PREV_RECENT_1Y_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_RECENT_2Y_RATIO"] = prev_agg["PREV_RECENT_2Y_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)

    prev_contract_types = pd.crosstab(previous["SK_ID_CURR"], previous["NAME_CONTRACT_TYPE"])
    prev_contract_types.columns = [f"PREV_CONTRACT_{str(c).replace(' ', '_')}" for c in prev_contract_types.columns]

    prev_product_types = pd.crosstab(previous["SK_ID_CURR"], previous["NAME_PORTFOLIO"])
    prev_product_types.columns = [f"PREV_PRODUCT_{str(c).replace(' ', '_')}" for c in prev_product_types.columns]

    prev_agg = prev_agg.join(prev_contract_types, how="left").join(prev_product_types, how="left")
    prev_agg = prev_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in prev_agg.columns:
        prev_agg[col] = prev_agg[col].astype(np.float32)

    del previous
    gc.collect()
    return prev_agg


# ----------------------------------------------------------------------
# 3. Installments aggregates
# ----------------------------------------------------------------------
def build_installments_features(path: str) -> pd.DataFrame:
    installments = pd.read_csv(path, encoding="latin-1")

    installments["DAYS_LATE"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["PAYMENT_RATIO"] = installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"].replace(0, np.nan)
    installments["LATE_PAYMENT"] = (installments["DAYS_LATE"] > 0).astype(np.int8)
    installments["SEVERE_LATE_PAYMENT"] = (installments["DAYS_LATE"] > 30).astype(np.int8)
    installments["UNDERPAYMENT"] = (installments["PAYMENT_RATIO"] < 0.95).astype(np.int8)
    installments["PAYMENT_RATIO_CAPPED"] = installments["PAYMENT_RATIO"].clip(0, 2)

    install_agg = installments.groupby("SK_ID_CURR").agg(
        INST_PAYMENT_COUNT=("SK_ID_PREV", "count"),
        INST_LATE_COUNT=("LATE_PAYMENT", "sum"),
        INST_SEVERE_LATE_COUNT=("SEVERE_LATE_PAYMENT", "sum"),
        INST_UNDERPAYMENT_COUNT=("UNDERPAYMENT", "sum"),
        INST_AVG_DAYS_LATE=("DAYS_LATE", "mean"),
        INST_MAX_DAYS_LATE=("DAYS_LATE", "max"),
        INST_AVG_PAYMENT_RATIO=("PAYMENT_RATIO_CAPPED", "mean"),
        INST_MIN_PAYMENT_RATIO=("PAYMENT_RATIO", "min"),
        INST_TOTAL_INSTALLMENT=("AMT_INSTALMENT", "sum"),
        INST_TOTAL_PAYMENT=("AMT_PAYMENT", "sum"),
        INST_AVG_INSTALLMENT=("AMT_INSTALMENT", "mean"),
        INST_AVG_PAYMENT=("AMT_PAYMENT", "mean"),
        INST_UNIQUE_PREV=("SK_ID_PREV", "nunique"),
    )

    install_agg["INST_LATE_RATIO"] = install_agg["INST_LATE_COUNT"] / install_agg["INST_PAYMENT_COUNT"].replace(0, np.nan)
    install_agg["INST_SEVERE_LATE_RATIO"] = install_agg["INST_SEVERE_LATE_COUNT"] / install_agg["INST_PAYMENT_COUNT"].replace(0, np.nan)
    install_agg["INST_UNDERPAYMENT_RATIO"] = install_agg["INST_UNDERPAYMENT_COUNT"] / install_agg["INST_PAYMENT_COUNT"].replace(0, np.nan)
    install_agg["INST_TOTAL_PAYMENT_RATIO"] = install_agg["INST_TOTAL_PAYMENT"] / install_agg["INST_TOTAL_INSTALLMENT"].replace(0, np.nan)

    install_agg = install_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in install_agg.columns:
        install_agg[col] = install_agg[col].astype(np.float32)

    del installments
    gc.collect()
    return install_agg


# ----------------------------------------------------------------------
# 4. POS-CASH aggregates
# ----------------------------------------------------------------------
def build_pos_cash_features(path: str) -> pd.DataFrame:
    pos = pd.read_csv(path, encoding="latin-1")

    pos["POS_DPD_FLAG"] = (pos["SK_DPD"] > 0).astype(np.int8)
    pos["POS_DPD_DEF_FLAG"] = (pos["SK_DPD_DEF"] > 0).astype(np.int8)
    pos["POS_SEVERE_DPD_FLAG"] = (pos["SK_DPD"] > 30).astype(np.int8)
    pos["POS_RECENT_1Y"] = (pos["MONTHS_BALANCE"] >= -12).astype(np.int8)
    pos["POS_RECENT_2Y"] = (pos["MONTHS_BALANCE"] >= -24).astype(np.int8)

    pos_agg = pos.groupby("SK_ID_CURR").agg(
        POS_RECORD_COUNT=("SK_ID_PREV", "count"),
        POS_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
        POS_DPD_COUNT=("POS_DPD_FLAG", "sum"),
        POS_DPD_DEF_COUNT=("POS_DPD_DEF_FLAG", "sum"),
        POS_SEVERE_DPD_COUNT=("POS_SEVERE_DPD_FLAG", "sum"),
        POS_AVG_DPD=("SK_DPD", "mean"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_AVG_DPD_DEF=("SK_DPD_DEF", "mean"),
        POS_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
        POS_AVG_INSTALLMENT=("CNT_INSTALMENT", "mean"),
        POS_AVG_INSTALLMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_MIN_INSTALLMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "min"),
        POS_MAX_INSTALLMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "max"),
        POS_MONTH_COUNT=("MONTHS_BALANCE", "nunique"),
        POS_RECENT_1Y_COUNT=("POS_RECENT_1Y", "sum"),
        POS_RECENT_2Y_COUNT=("POS_RECENT_2Y", "sum"),
    )

    denom = pos_agg["POS_RECORD_COUNT"].replace(0, np.nan)
    pos_agg["POS_DPD_RATIO"] = pos_agg["POS_DPD_COUNT"] / denom
    pos_agg["POS_DPD_DEF_RATIO"] = pos_agg["POS_DPD_DEF_COUNT"] / denom
    pos_agg["POS_SEVERE_DPD_RATIO"] = pos_agg["POS_SEVERE_DPD_COUNT"] / denom
    pos_agg["POS_RECENT_1Y_RATIO"] = pos_agg["POS_RECENT_1Y_COUNT"] / denom
    pos_agg["POS_RECENT_2Y_RATIO"] = pos_agg["POS_RECENT_2Y_COUNT"] / denom

    status_counts = pd.crosstab(pos["SK_ID_CURR"], pos["NAME_CONTRACT_STATUS"])
    status_counts.columns = ["POS_STATUS_" + str(c).replace(" ", "_") for c in status_counts.columns]
    pos_agg = pos_agg.join(status_counts, how="left")
    for col in status_counts.columns:
        pos_agg[col + "_RATIO"] = pos_agg[col] / pos_agg["POS_RECORD_COUNT"].replace(0, np.nan)

    pos_agg = pos_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in pos_agg.columns:
        pos_agg[col] = pos_agg[col].astype(np.float32)

    del pos
    gc.collect()
    return pos_agg


# ----------------------------------------------------------------------
# 5. Bureau (credit bureau) aggregates
# ----------------------------------------------------------------------
def build_bureau_features(bureau_path: str, bb_path: str, base_ids: pd.DataFrame):
    bb = pd.read_csv(bb_path, encoding="latin-1")
    status_num = pd.to_numeric(bb["STATUS"], errors="coerce")
    bb["BB_DPD_FLAG"] = (status_num.fillna(0) > 0).astype("int8")
    bb["BB_CLOSED_FLAG"] = (bb["STATUS"] == "C").astype("int8")

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        BB_DPD_RATIO=("BB_DPD_FLAG", "mean"),
        BB_CLOSED_RATIO=("BB_CLOSED_FLAG", "mean"),
        BB_MAX_STATUS=("STATUS", lambda x: pd.to_numeric(x, errors="coerce").max()),
    ).reset_index()
    del bb
    gc.collect()

    bureau = pd.read_csv(bureau_path, encoding="latin-1")
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb_agg
    gc.collect()

    bureau["BUREAU_ACTIVE_FLAG"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")
    bureau["BUREAU_CLOSED_FLAG"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype("int8")
    bureau["BUREAU_OVERDUE_FLAG"] = (bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0).astype("int8")
    bureau["BUREAU_RECENT_1Y"] = (bureau["DAYS_CREDIT"] >= -365).astype("int8")
    bureau["BUREAU_RECENT_2Y"] = (bureau["DAYS_CREDIT"] >= -730).astype("int8")

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_RATIO=("BUREAU_ACTIVE_FLAG", "mean"),
        BUREAU_CLOSED_RATIO=("BUREAU_CLOSED_FLAG", "mean"),
        BUREAU_OVERDUE_RATIO=("BUREAU_OVERDUE_FLAG", "mean"),
        BUREAU_TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
        BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_TOTAL_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_MAX_DAYS_CREDIT_ENDDATE=("DAYS_CREDIT_ENDDATE", "max"),
        BUREAU_RECENT_1Y_COUNT=("BUREAU_RECENT_1Y", "sum"),
        BUREAU_RECENT_2Y_COUNT=("BUREAU_RECENT_2Y", "sum"),
        BUREAU_AVG_DPD_RATIO=("BB_DPD_RATIO", "mean"),
        BUREAU_MAX_STATUS=("BB_MAX_STATUS", "max"),
        BUREAU_AVG_MONTH_COUNT=("BB_MONTH_COUNT", "mean"),
    )
    bureau_agg["BUREAU_DEBT_TO_CREDIT"] = bureau_agg["BUREAU_TOTAL_DEBT"] / bureau_agg["BUREAU_TOTAL_CREDIT"].replace(0, np.nan)
    bureau_agg = bureau_agg.reset_index()

    del bureau
    gc.collect()

    bureau_merged = base_ids.merge(bureau_agg, on="SK_ID_CURR", how="left")
    bureau_numeric_cols = [c for c in bureau_agg.columns if c != "SK_ID_CURR"]
    for col in BUREAU_MISSING_FLAG_COLS:
        bureau_merged[f"{col}_MISSING"] = bureau_merged[col].isna().astype("int8")
    bureau_merged = bureau_merged.fillna(0)

    bureau_feature_cols_final = [
        c for c in (bureau_numeric_cols + [f"{c}_MISSING" for c in BUREAU_MISSING_FLAG_COLS])
        if c not in BUREAU_DROP_COLS
    ]

    X_bureau_final = sparse.csr_matrix(bureau_merged[bureau_feature_cols_final].astype(np.float32).values)
    return X_bureau_final, bureau_feature_cols_final, bureau_agg


# ----------------------------------------------------------------------
# 6. Credit card balance aggregates
# ----------------------------------------------------------------------
def build_credit_card_features(cc_path: str, base_ids: pd.DataFrame):
    cc = pd.read_csv(cc_path, encoding="latin-1")

    cc["CC_UTILIZATION"] = np.where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0, cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"], np.nan
    )
    cc["CC_DPD_FLAG"] = (cc["SK_DPD"] > 0).astype("int8")
    cc["CC_DPD_DEF_FLAG"] = (cc["SK_DPD_DEF"] > 0).astype("int8")
    cc["CC_MIN_PAYMENT_RATIO"] = np.where(
        cc["AMT_INST_MIN_REGULARITY"] > 0,
        cc["AMT_PAYMENT_CURRENT"].fillna(0) / cc["AMT_INST_MIN_REGULARITY"],
        np.nan,
    )
    cc["CC_DRAWING_ACTIVITY"] = (cc["AMT_DRAWINGS_CURRENT"].fillna(0) > 0).astype("int8")
    cc["CC_RECENT_1Y"] = (cc["MONTHS_BALANCE"] >= -12).astype("int8")

    cc_agg = cc.groupby("SK_ID_CURR").agg(
        CC_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        CC_AVG_UTILIZATION=("CC_UTILIZATION", "mean"),
        CC_MAX_UTILIZATION=("CC_UTILIZATION", "max"),
        CC_AVG_BALANCE=("AMT_BALANCE", "mean"),
        CC_MAX_BALANCE=("AMT_BALANCE", "max"),
        CC_AVG_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DPD_RATIO=("CC_DPD_FLAG", "mean"),
        CC_DPD_DEF_RATIO=("CC_DPD_DEF_FLAG", "mean"),
        CC_MAX_DPD=("SK_DPD", "max"),
        CC_AVG_MIN_PAYMENT_RATIO=("CC_MIN_PAYMENT_RATIO", "mean"),
        CC_DRAWING_RATIO=("CC_DRAWING_ACTIVITY", "mean"),
        CC_TOTAL_DRAWINGS=("AMT_DRAWINGS_CURRENT", "sum"),
        CC_RECENT_1Y_COUNT=("CC_RECENT_1Y", "sum"),
        CC_UNIQUE_PREV=("SK_ID_PREV", "nunique"),
    ).reset_index()

    del cc
    gc.collect()

    cc_merged = base_ids.merge(cc_agg, on="SK_ID_CURR", how="left")
    cc_numeric_cols = [c for c in cc_agg.columns if c != "SK_ID_CURR"]
    for col in CC_RATIO_MISSING_FLAG_COLS:
        cc_merged[f"{col}_MISSING"] = cc_merged[col].isna().astype("int8")
    cc_merged = cc_merged.fillna(0)

    cc_feature_cols_final = [
        c for c in (cc_numeric_cols + [f"{c}_MISSING" for c in CC_RATIO_MISSING_FLAG_COLS])
        if c not in CC_DROP_COLS
    ]

    X_cc_final = sparse.csr_matrix(cc_merged[cc_feature_cols_final].astype(np.float32).values)
    return X_cc_final, cc_feature_cols_final, cc_agg


# ----------------------------------------------------------------------
# 7. Assemble the application-level block (X_pos): app + prev + install + pos
# ----------------------------------------------------------------------
def assemble_application_block(df_fe: pd.DataFrame, prev_agg: pd.DataFrame, install_agg: pd.DataFrame, pos_agg: pd.DataFrame):
    df_model = df_fe.merge(prev_agg, left_on="SK_ID_CURR", right_index=True, how="left")
    df_model = df_model.merge(install_agg[INSTALL_SELECTED], left_on="SK_ID_CURR", right_index=True, how="left")
    df_model = df_model.merge(pos_agg[POS_SELECTED], left_on="SK_ID_CURR", right_index=True, how="left")

    y = df_model["TARGET"].astype(np.int8) if "TARGET" in df_model.columns else None
    drop_cols = [c for c in ["TARGET", "SK_ID_CURR"] if c in df_model.columns]
    X_df = df_model.drop(columns=drop_cols)

    categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    X_num = X_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    X_cat = X_df[categorical_cols].fillna("Missing").astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    X_cat_encoded = encoder.fit_transform(X_cat)

    X_num_sparse = sparse.csr_matrix(X_num.values, dtype=np.float32)
    X_pos = sparse.hstack([X_num_sparse, X_cat_encoded], format="csr", dtype=np.float32)

    feature_names = numeric_cols + encoder.get_feature_names_out(categorical_cols).tolist()

    del df_model, X_df, X_num, X_cat, X_cat_encoded, X_num_sparse
    gc.collect()

    return X_pos, feature_names, encoder, numeric_cols, categorical_cols, y
