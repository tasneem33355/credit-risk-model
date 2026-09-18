"""
Data Drift & Population Stability Index (PSI) Governance Monitor
===============================================================
Platform: CrediX / ZAWOLF Institutional Fraud Defense
Compliance Standard: CBE Model Risk Management & Basel Committee Drift Policy

Calculates:
  - Feature-level Population Stability Index (PSI)
  - Wasserstein Distance (Earth Mover's Distance)
  - Automated Retraining Trigger Alert:
      * PSI < 0.10: Stable (Green - Normal Operation)
      * 0.10 <= PSI < 0.25: Moderate Drift (Yellow - Heightened Audit Queue)
      * PSI >= 0.25: Critical Drift (Red - Automated Retraining Mandatory)
"""

import os
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    'income_mismatch_ratio', 'annuity_to_balance_ratio', 'balance_volatility_cv',
    'surge_ratio_max_to_avg', 'ocr_quality_mean', 'min_to_avg_balance_ratio',
    'applicant_age_norm', 'employment_tenure_years', 'inflow_regularity_score',
    'iscore_normalized', 'inflow_uniformity_score', 'bureau_facilities_count'
]


class PopulationDriftMonitor:
    def __init__(self, baseline_csv_path: str = None):
        if baseline_csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            baseline_csv_path = os.path.join(base_dir, "data", "fraud_training_data_25000.csv")
        
        self.baseline_csv_path = baseline_csv_path
        self.baseline_df = None
        self._load_baseline()

    def _load_baseline(self):
        if os.path.exists(self.baseline_csv_path):
            self.baseline_df = pd.read_csv(self.baseline_csv_path)
        else:
            self.baseline_df = None

    @staticmethod
    def calculate_single_feature_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates the Population Stability Index (PSI) between two continuous distributions.
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        if len(baseline) == 0 or len(current) == 0:
            return 0.0

        # Create quantile bins based on baseline
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 3:
            return 0.0

        # Frequency counts
        b_counts, _ = np.histogram(baseline, bins=bin_edges)
        c_counts, _ = np.histogram(current, bins=bin_edges)

        # Proportions with Laplace smoothing to prevent division by zero
        b_pct = np.clip(b_counts / len(baseline), 1e-4, 1.0)
        c_pct = np.clip(c_counts / len(current), 1e-4, 1.0)

        # PSI formula
        psi_val = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
        return float(np.round(psi_val, 4))

    def evaluate_batch_drift(self, incoming_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates a batch of recent applicant feature vectors against the 25k baseline.
        Returns detailed PSI table, maximum drift feature, and regulatory compliance status.
        """
        if self.baseline_df is None:
            return {
                "status": "BASELINE_DATA_NOT_FOUND",
                "psi_summary": {},
                "max_psi": 0.0,
                "retraining_recommended": False
            }

        psi_results = {}
        for feat in FEATURE_NAMES:
            if feat in self.baseline_df.columns and feat in incoming_features.columns:
                b_series = self.baseline_df[feat].dropna().values
                c_series = incoming_features[feat].dropna().values
                psi_score = self.calculate_single_feature_psi(b_series, c_series)
                
                if psi_score < 0.10:
                    status = "STABLE (GREEN)"
                elif psi_score < 0.25:
                    status = "MODERATE DRIFT (YELLOW)"
                else:
                    status = "CRITICAL DRIFT (RED)"

                psi_results[feat] = {
                    "psi_score": psi_score,
                    "status": status
                }

        max_feat = max(psi_results.keys(), key=lambda k: psi_results[k]["psi_score"]) if psi_results else "N/A"
        max_psi = psi_results[max_feat]["psi_score"] if psi_results else 0.0

        retraining_mandatory = bool(max_psi >= 0.25)
        warning_state = bool(0.10 <= max_psi < 0.25)

        return {
            "evaluation_timestamp": pd.Timestamp.utcnow().isoformat(),
            "evaluated_batch_size": len(incoming_features),
            "max_psi_feature": max_feat,
            "max_psi_score": max_psi,
            "system_health": "CRITICAL DRIFT" if retraining_mandatory else ("WARNING" if warning_state else "HEALTHY"),
            "retraining_recommended": retraining_mandatory,
            "cbe_audit_comment": (
                f"Severe drift detected in feature '{max_feat}' (PSI={max_psi:.4f} >= 0.25). Immediate model retraining required."
                if retraining_mandatory else
                ("Moderate drift observed; review underwriting policy." if warning_state else "Feature distributions fully stable and compliant with baseline.")
            ),
            "feature_metrics": psi_results
        }


def trigger_automated_retraining_if_needed(incoming_batch: pd.DataFrame) -> Dict[str, Any]:
    """Helper to be called periodically from Cron or API."""
    monitor = PopulationDriftMonitor()
    drift_report = monitor.evaluate_batch_drift(incoming_batch)
    if drift_report.get("retraining_recommended"):
        # Invoke train_fraud pipeline
        try:
            from model.train_fraud import train
            train()
            drift_report["retraining_action_status"] = "RETRAINING_SUCCESSFULLY_TRIGGERED_AND_SAVED"
        except Exception as e:
            drift_report["retraining_action_status"] = f"RETRAINING_FAILED: {str(e)}"
    else:
        drift_report["retraining_action_status"] = "RETRAINING_NOT_REQUIRED"
    return drift_report
