"""
End-to-end retraining script for the credit risk model.

Reproduces the pipeline developed in notebooks/credit_risk_model.ipynb:
    application_train + previous_application + installments_payments +
    POS_CASH_balance + bureau + bureau_balance + credit_card_balance
    -> 413-feature sparse matrix -> XGBoost + LightGBM ensemble.

Usage:
    python model/train.py --data-dir /path/to/home-credit-default-risk \
                           --out-dir model/artifacts

Produces in --out-dir:
    xgb_final.pkl        trained XGBoost model
    lgb_final.pkl        trained LightGBM model
    encoder.pkl          fitted OneHotEncoder for application categoricals
    feature_names.pkl    ordered list of the 413 final feature names
    metrics.json         ROC-AUC, PR-AUC, Gini, K-S, blend weights, cutoffs
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb

from features import (
    add_core_features,
    add_secondary_features,
    build_previous_application_features,
    build_installments_features,
    build_pos_cash_features,
    build_bureau_features,
    build_credit_card_features,
    assemble_application_block,
)


def build_dataset(data_dir: str):
    app_path = os.path.join(data_dir, "application_train.csv")
    df_app = pd.read_csv(app_path, encoding="latin-1")
    print(f"application_train: {df_app.shape}, TARGET rate: {df_app['TARGET'].mean():.4%}")

    df_fe = add_core_features(df_app)
    df_fe = add_secondary_features(df_fe)

    prev_agg = build_previous_application_features(os.path.join(data_dir, "previous_application.csv"))
    install_agg = build_installments_features(os.path.join(data_dir, "installments_payments.csv"))
    pos_agg = build_pos_cash_features(os.path.join(data_dir, "POS_CASH_balance.csv"))

    X_pos, feature_names, encoder, numeric_cols, categorical_cols, y = assemble_application_block(
        df_fe, prev_agg, install_agg, pos_agg
    )

    base_ids = df_fe[["SK_ID_CURR"]].copy()
    X_bureau_final, bureau_feature_cols_final, bureau_agg = build_bureau_features(
        os.path.join(data_dir, "bureau.csv"), os.path.join(data_dir, "bureau_balance.csv"), base_ids
    )
    X_cc_final, cc_feature_cols_final, cc_agg = build_credit_card_features(
        os.path.join(data_dir, "credit_card_balance.csv"), base_ids
    )

    X_final = sparse.hstack([X_pos, X_bureau_final, X_cc_final], format="csr")
    feature_names_final = feature_names + bureau_feature_cols_final + cc_feature_cols_final
    assert X_final.shape[1] == len(feature_names_final), "Feature name / column count mismatch"

    print(f"Final matrix shape: {X_final.shape}")

    feature_schema = {
        "feature_names": feature_names_final,
        "application_numeric_cols": numeric_cols,
        "application_categorical_cols": categorical_cols,
        "bureau_feature_cols": bureau_feature_cols_final,
        "cc_feature_cols": cc_feature_cols_final,
    }

    del df_fe, X_pos, X_bureau_final, X_cc_final
    gc.collect()

    return X_final, feature_names_final, encoder, y, feature_schema


def train_models(X_final, y):
    train_idx, valid_idx = train_test_split(
        np.arange(X_final.shape[0]), test_size=0.20, stratify=y, random_state=42
    )
    X_tr, X_va = X_final[train_idx], X_final[valid_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
    print(f"Train shape: {X_tr.shape} | Valid shape: {X_va.shape}")

    xgb_final = XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.05,
        subsample=0.70, colsample_bytree=0.80, min_child_weight=10, gamma=0.10,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    xgb_final.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_xgb = xgb_final.predict_proba(X_va)[:, 1]
    print(f"XGBoost  -> ROC-AUC: {roc_auc_score(y_va, pred_xgb):.5f}")

    lgb_final = lgb.LGBMClassifier(
        n_estimators=1500, max_depth=4, num_leaves=15, learning_rate=0.03,
        subsample=0.70, colsample_bytree=0.80, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, objective="binary",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_final.fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_lgb = lgb_final.predict_proba(X_va)[:, 1]
    print(f"LightGBM -> ROC-AUC: {roc_auc_score(y_va, pred_lgb):.5f}")

    blend_results = []
    for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_lgb = 1 - w_xgb
        pred_blend = w_xgb * pred_xgb + w_lgb * pred_lgb
        blend_results.append({"w_xgb": w_xgb, "w_lgb": w_lgb, "ROC_AUC": roc_auc_score(y_va, pred_blend)})
    blend_df = pd.DataFrame(blend_results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    best_w_xgb = float(blend_df.iloc[0]["w_xgb"])
    best_w_lgb = float(blend_df.iloc[0]["w_lgb"])
    pred_final_blend = best_w_xgb * pred_xgb + best_w_lgb * pred_lgb

    final_roc_auc = roc_auc_score(y_va, pred_final_blend)
    final_pr_auc = average_precision_score(y_va, pred_final_blend)
    gini = 2 * final_roc_auc - 1
    ks_stat, _ = ks_2samp(pred_final_blend[y_va.values == 0], pred_final_blend[y_va.values == 1])

    print(f"\n>>> FINAL BLEND (w_xgb={best_w_xgb}, w_lgb={best_w_lgb}) "
          f"-> ROC-AUC: {final_roc_auc:.5f} | PR-AUC: {final_pr_auc:.5f} "
          f"| Gini: {gini*100:.2f}% | K-S: {ks_stat*100:.2f}%")

    metrics = {
        "roc_auc": final_roc_auc,
        "pr_auc": final_pr_auc,
        "gini": gini,
        "ks_statistic": float(ks_stat),
        "baseline_bad_rate": float(y_va.mean()),
        "blend_weights": {"xgb": best_w_xgb, "lgb": best_w_lgb},
        "n_train": int(X_tr.shape[0]),
        "n_valid": int(X_va.shape[0]),
        "n_features": int(X_final.shape[1]),
        # Operational cutoffs on predicted probability of default (PD):
        #   PD < cutoff_approve            -> AUTO-APPROVE
        #   cutoff_approve <= PD < cutoff_reject -> MANUAL REVIEW (collateral / guarantor)
        #   PD >= cutoff_reject             -> AUTO-REJECT
        # These are policy choices tuned against the portfolio simulation in
        # docs/model_card.md, not statistical outputs -- revisit whenever the
        # bank's risk appetite or NPL target changes.
        "cutoff_approve": 0.0723,
        "cutoff_reject": 0.20,
    }
    return xgb_final, lgb_final, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with the Home Credit CSV files")
    parser.add_argument("--out-dir", default="model/artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    X_final, feature_names, encoder, y, feature_schema = build_dataset(args.data_dir)
    xgb_final, lgb_final, metrics = train_models(X_final, y)

    with open(os.path.join(args.out_dir, "xgb_final.pkl"), "wb") as f:
        pickle.dump(xgb_final, f)
    with open(os.path.join(args.out_dir, "lgb_final.pkl"), "wb") as f:
        pickle.dump(lgb_final, f)
    with open(os.path.join(args.out_dir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(args.out_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    with open(os.path.join(args.out_dir, "feature_schema.json"), "w") as f:
        json.dump(feature_schema, f, indent=2)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to {args.out_dir} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
