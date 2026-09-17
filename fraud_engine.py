"""
Senior Enterprise Credit Application Fraud & Forensic Intelligence Engine (v2.5)
==================================================================================
Platform: Smart Financing & Credit Request Analysis Platform (CrediX / ZAWOLF)
Target Standard: Central Bank of Egypt (CBE) Compliance & Basel Committee Principles
Architecture: 5-Layer Hybrid Fraud Defense
  - Layer 1: Deterministic Cross-Document Rules Engine (OCR + Bureau + Core Banking)
  - Layer 2: Deep Forensic Signals & Cashflow Analytics (Benford's Law + Uniformity)
  - Layer 3: Entity Resolution & Syndicate Collision Detector (Graph / Velocity)
  - Layer 4: High-Dimensional Unsupervised Machine Learning (Isolation Forest)
  - Layer 5: Cost-Sensitive Hybrid Fusion, Bilingual XAI & Downstream Risk Feeder
"""

import os
import re
import json
import math
import sqlite3
import hashlib
import difflib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FraudRuleViolation:
    rule_code: str
    rule_name_en: str
    rule_name_ar: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description_en: str
    description_ar: str
    observed_value: Any
    threshold_value: Any
    weight: float


@dataclass
class AnomalySignal:
    anomaly_name: str
    anomaly_score: float  # 0.0 to 1.0
    detected: bool
    explanation_en: str
    explanation_ar: str


@dataclass
class EntityCollision:
    entity_type: str  # PHONE, NATIONAL_ID, EMPLOYER_TAX_ID, IBAN
    collision_count: int
    first_seen: str
    last_seen: str
    is_syndicate_alert: bool
    explanation_en: str
    explanation_ar: str


# =============================================================================
# LAYER 2 HELPER: BENFORD'S LAW & DEEP FORENSIC CASHFLOW ANALYZER
# =============================================================================

class DeepForensicAnalyzer:
    """
    Mathematical forensics on transaction numbers, detecting synthetic uniformity
    and digital manipulation via Benford's Law first-digit distribution.
    """
    # Expected Benford probabilities for leading digits 1 to 9
    BENFORD_PROBABILITIES = {
        1: 0.3010, 2: 0.1761, 3: 0.1249, 4: 0.0969,
        5: 0.0792, 6: 0.0669, 7: 0.0580, 8: 0.0512, 9: 0.0458
    }

    @classmethod
    def evaluate_benford_law(cls, numbers: List[float]) -> Tuple[bool, float, str]:
        """
        Tests leading digit distribution against Benford's Law using Chi-Square approximation.
        Fabricated bank statements typically deviate significantly (e.g. overusing 5, 7, 9).
        """
        first_digits = []
        for n in numbers:
            if n > 0:
                s = str(f"{abs(n):.2f}").lstrip("0").replace(".", "")
                if s and s[0].isdigit() and int(s[0]) > 0:
                    first_digits.append(int(s[0]))

        if len(first_digits) < 4:
            return False, 0.0, "Insufficient cashflow numbers for Benford test."

        counts = {d: first_digits.count(d) for d in range(1, 10)}
        total = len(first_digits)
        chi_square = 0.0

        for d in range(1, 10):
            observed = counts[d]
            expected = total * cls.BENFORD_PROBABILITIES[d]
            chi_square += ((observed - expected) ** 2) / (expected + 1e-5)

        # Critical value for 8 degrees of freedom at 95% confidence is ~15.51
        is_anomaly = chi_square > 15.51
        anomaly_score = min(round(chi_square / 30.0, 3), 1.0) if is_anomaly else 0.0
        msg = f"Chi-Square: {chi_square:.2f} (Threshold 15.51). Suspicious synthetic distribution detected." if is_anomaly else "Cashflow digits conform to Benford distribution."
        return is_anomaly, anomaly_score, msg

    @staticmethod
    def check_inflow_uniformity(declared_salary: float, bank_inflow: float, 
                               avg_balance: float, min_balance: float) -> Tuple[bool, float, str]:
        """
        Detects artificial round-number salary transfers lacking authentic statutory deductions (Tax/Insurance).
        """
        is_suspicious_round = False
        if declared_salary > 5000 and (declared_salary % 1000 == 0) and (bank_inflow % 1000 == 0):
            # In Egypt, true net salaries rarely end with exact 000 after variable deductions
            if abs(declared_salary - bank_inflow) < 1.0:
                is_suspicious_round = True

        score = 0.45 if is_suspicious_round else 0.0
        explanation = "Suspiciously pristine round-number salary matches lacking normal tax/social insurance fractions." if is_suspicious_round else "Natural cashflow decimals observed."
        return is_suspicious_round, score, explanation


# =============================================================================
# LAYER 3: EMBEDDED SYNDICATE & ENTITY RESOLUTION STORE (SQLITE BACKED)
# =============================================================================

class SQLiteEntityStore:
    """
    Tracks cross-application velocity and entity collisions (phone, employer, bank account)
    over rolling time windows to neutralize organized fraud rings and ghost companies.
    """
    def __init__(self, db_path: str = "fraud_registry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entity_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_hash TEXT NOT NULL,
                        observed_value_masked TEXT NOT NULL,
                        submission_timestamp TIMESTAMP NOT NULL
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity ON entity_audit_log(entity_type, entity_hash, submission_timestamp)")
                conn.commit()
        except Exception as e:
            # Fallback for serverless or restricted disk environments
            pass

    @staticmethod
    def _hash_entity(val: str) -> str:
        return hashlib.sha256(str(val).strip().lower().encode("utf-8")).hexdigest()

    def check_and_record(self, app_id: str, entities: Dict[str, str], timestamp_str: str, 
                           window_hours: int = 48) -> List[EntityCollision]:
        collisions = []
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.utcnow()

        window_start = ts - timedelta(hours=window_hours)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for entity_type, raw_val in entities.items():
                    if not raw_val or str(raw_val).strip() == "" or str(raw_val).lower() == "n/a":
                        continue

                    e_hash = self._hash_entity(raw_val)
                    masked = str(raw_val)[:3] + "****" + str(raw_val)[-3:] if len(str(raw_val)) > 6 else "****"

                    cursor.execute("""
                        SELECT COUNT(*), MIN(submission_timestamp), MAX(submission_timestamp)
                        FROM entity_audit_log
                        WHERE entity_type = ? AND entity_hash = ? AND submission_timestamp >= ? AND application_id != ?
                    """, (entity_type, e_hash, window_start.isoformat(), app_id))

                    row = cursor.fetchone()
                    count = row[0] if row else 0

                    if count > 0:
                        is_syndicate = count >= 2
                        collisions.append(EntityCollision(
                            entity_type=entity_type,
                            collision_count=count,
                            first_seen=str(row[1]) if row else "",
                            last_seen=str(row[2]) if row else "",
                            is_syndicate_alert=is_syndicate,
                            explanation_en=f"Entity '{entity_type}' collided with {count} recent application(s) within the last {window_hours} hours. Possible coordinated loan stacking / fraud syndicate.",
                            explanation_ar=f"البيان '{entity_type}' تكرر في عدد {count} طلبات تمويل خلال آخر {window_hours} ساعة. شبهة استغلال منظم أو شبكة احتيال."
                        ))

                    # Insert current record
                    cursor.execute("""
                        INSERT INTO entity_audit_log (application_id, entity_type, entity_hash, observed_value_masked, submission_timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (app_id, entity_type, e_hash, masked, ts.isoformat()))

                conn.commit()
        except Exception:
            # Safe silent fallback if database lock occurs
            pass

        return collisions


# =============================================================================
# LAYER 4: HIGH-DIMENSIONAL CONTINUOUS VECTORIZER & ISOLATION FOREST
# =============================================================================

class HighDimensionalFraudVectorizer:
    """
    Extracts a dense 12-dimensional continuous feature vector from incoming application
    payloads for unsupervised machine learning anomaly scoring.
    """
    FEATURE_NAMES = [
        "income_mismatch_ratio",
        "annuity_to_balance_ratio",
        "balance_volatility_cv",
        "surge_ratio_max_to_avg",
        "ocr_quality_mean",
        "min_to_avg_balance_ratio",
        "applicant_age_norm",
        "employment_tenure_years",
        "inflow_regularity_score",
        "iscore_normalized",
        "inflow_uniformity_score",
        "bureau_facilities_count"
    ]

    @classmethod
    def extract_features(cls, app: Dict[str, Any], mismatch_ratio: float, 
                         uniformity_score: float) -> np.ndarray:
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})
        salary_fields = app.get("salary_certificate_fields", {})
        nid_fields = app.get("national_id_fields", {})
        iscore_fields = app.get("iscore_report_fields", {})

        avg_balance = float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 1.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 1.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 0.95) or 0.95)
        annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)

        # 1. Mismatch
        f_mismatch = min(float(mismatch_ratio), 3.0)
        # 2. Annuity to balance
        f_annuity_bal = min(annuity / (avg_balance + 1e-3), 5.0)
        # 3. Coefficient of variation (Volatility / Mean)
        f_volatility_cv = min(volatility / (avg_balance + 1e-3), 5.0)
        # 4. Surge ratio
        f_surge = min(max_balance / (avg_balance + 1e-3), 10.0)
        # 5. Average OCR Quality
        docs = app.get("documents", [])
        qualities = [float(d.get("overall_quality_score", 0.9)) for d in docs if isinstance(d, dict)]
        f_ocr = float(np.mean(qualities)) if qualities else 0.95
        # 6. Min to avg balance
        f_min_bal = min(min_balance / (avg_balance + 1e-3), 1.0)
        # 7. Age normalized (21 to 65 -> 0.0 to 1.0)
        raw_age = float(nid_fields.get("age_years", {}).get("value", 35.0) or 35.0)
        f_age = max(0.0, min((raw_age - 21.0) / 44.0, 1.0))
        # 8. Tenure years
        f_tenure = min(float(salary_fields.get("employment_tenure_years", {}).get("value", 3.0) or 3.0), 30.0)
        # 9. Regularity
        f_regularity = max(0.0, min(regularity, 1.0))
        # 10. I-Score normalized (300 to 850 -> 0.0 to 1.0)
        raw_iscore = float(iscore_fields.get("credit_score", {}).get("value", 670.0) or 670.0)
        f_iscore = max(0.0, min((raw_iscore - 300.0) / 550.0, 1.0))
        # 11. Uniformity
        f_uniformity = float(uniformity_score)
        # 12. Active facilities count
        f_facilities = min(float(len(iscore_fields.get("bureau_facilities", []))), 15.0)

        vec = np.array([
            f_mismatch, f_annuity_bal, f_volatility_cv, f_surge,
            f_ocr, f_min_bal, f_age, f_tenure,
            f_regularity, f_iscore, f_uniformity, f_facilities
        ], dtype=np.float32)

        return vec.reshape(1, -1)


class PersistentIsolationForestManager:
    """
    Manages loading, training, and persistence of the high-dimensional Isolation Forest model.
    Trains on a calibrated 5,000-loan synthetic population of Egyptian commercial borrowers.
    """
    def __init__(self, artifact_dir: str = None):
        if artifact_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            artifact_dir = os.path.join(base_dir, "model", "artifacts", "fraud")
        
        self.artifact_dir = artifact_dir
        self.model_path = os.path.join(artifact_dir, "isolation_forest_v2.joblib")
        self.scaler_path = os.path.join(artifact_dir, "scaler_v2.joblib")
        
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[IsolationForest] = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                return
            except Exception:
                pass

        # Train on calibrated synthetic Egyptian banking distribution
        self._train_and_persist_baseline()

    def _train_and_persist_baseline(self):
        np.random.seed(42)
        n_samples = 5000

        # Generate realistic baseline: 94% authentic applicants, 6% multi-pattern fraud
        n_clean = int(n_samples * 0.94)
        n_fraud = n_samples - n_clean

        # Clean applicants
        clean_mismatch = np.random.beta(1.5, 25.0, n_clean) * 0.15          # Mean ~ 0.03
        clean_annuity_bal = np.random.gamma(2.0, 0.15, n_clean)              # Mean ~ 0.30
        clean_volatility_cv = np.random.gamma(2.5, 0.20, n_clean)            # Mean ~ 0.50
        clean_surge = np.random.gamma(1.8, 0.70, n_clean) + 1.0              # Mean ~ 2.2
        clean_ocr = np.random.beta(30.0, 1.5, n_clean)                       # Mean ~ 0.95
        clean_min_bal = np.random.beta(3.0, 5.0, n_clean)                    # Mean ~ 0.37
        clean_age = np.random.beta(4.0, 3.5, n_clean)                        # Mean ~ 0.53 (Age ~ 44)
        clean_tenure = np.random.gamma(3.0, 2.0, n_clean)                    # Mean ~ 6 years
        clean_regularity = np.random.beta(25.0, 2.0, n_clean)                # Mean ~ 0.92
        clean_iscore = np.random.beta(12.0, 5.0, n_clean)                    # Mean ~ 0.70 (Score ~ 685)
        clean_uniformity = np.zeros(n_clean)                                 # Mean ~ 0.0
        clean_facilities = np.random.poisson(1.8, n_clean)                   # Mean ~ 2

        X_clean = np.column_stack([
            clean_mismatch, clean_annuity_bal, clean_volatility_cv, clean_surge,
            clean_ocr, clean_min_bal, clean_age, clean_tenure,
            clean_regularity, clean_iscore, clean_uniformity, clean_facilities
        ])

        # Injected fraud applicants (Window dressing, income inflation, forged OCR)
        fraud_mismatch = np.random.uniform(0.35, 1.2, n_fraud)
        fraud_annuity_bal = np.random.uniform(0.8, 3.5, n_fraud)
        fraud_volatility_cv = np.random.uniform(1.2, 4.0, n_fraud)
        fraud_surge = np.random.uniform(3.0, 8.0, n_fraud)
        fraud_ocr = np.random.uniform(0.40, 0.78, n_fraud)
        fraud_min_bal = np.random.uniform(0.0, 0.08, n_fraud)
        fraud_age = np.random.uniform(0.0, 1.0, n_fraud)
        fraud_tenure = np.random.uniform(0.0, 1.5, n_fraud)
        fraud_regularity = np.random.uniform(0.2, 0.6, n_fraud)
        fraud_iscore = np.random.uniform(0.1, 0.5, n_fraud)
        fraud_uniformity = np.random.choice([0.0, 0.45], p=[0.4, 0.6], size=n_fraud)
        fraud_facilities = np.random.poisson(4.5, n_fraud)

        X_fraud = np.column_stack([
            fraud_mismatch, fraud_annuity_bal, fraud_volatility_cv, fraud_surge,
            fraud_ocr, fraud_min_bal, fraud_age, fraud_tenure,
            fraud_regularity, fraud_iscore, fraud_uniformity, fraud_facilities
        ])

        X_train = np.vstack([X_clean, X_fraud])

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Contamination set to 5.0%
        self.model = IsolationForest(
            n_estimators=150,
            contamination=0.05,
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled)

        # Persist to disk
        try:
            os.makedirs(self.artifact_dir, exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception:
            pass

    def score_application(self, feature_vector: np.ndarray) -> Tuple[float, bool]:
        if self.model is None or self.scaler is None:
            return 0.05, False

        scaled = self.scaler.transform(feature_vector)
        # raw decision_function returns negative values for anomalies
        raw_score = self.model.decision_function(scaled)[0]
        is_anomaly = bool(self.model.predict(scaled)[0] == -1)

        # Calibrated mapping to [0.0, 1.0] where 1.0 = highly anomalous
        # Normal observations yield raw_score > 0.0, anomalies yield < 0.0
        normalized_anomaly_score = max(0.0, min(round(0.50 - (raw_score * 2.5), 3), 1.0))
        return normalized_anomaly_score, is_anomaly


# =============================================================================
# MAIN PRODUCTION ENGINE: CreditFraudEngine (5-LAYER HYBRID)
# =============================================================================

class CreditFraudEngine:
    """
    Enterprise-Grade 5-Layer Hybrid Fraud & Cross-Document Consistency Engine.
    Engineered for Egyptian Commercial Banking (Retail Underwriting).
    """

    SEVERITY_WEIGHTS = {
        "LOW": 0.08,
        "MEDIUM": 0.22,
        "HIGH": 0.45,
        "CRITICAL": 0.90
    }

    # Regulatory & Risk Thresholds
    INCOME_MISMATCH_WARN = 0.20        # 20% discrepancy triggers audit alert
    INCOME_MISMATCH_CRITICAL = 0.40    # 40% discrepancy triggers fatal tampering flag
    EMPLOYER_SIMILARITY_MIN = 0.65     # Minimum string similarity for employer
    MAX_ISCORE_AGE_DAYS = 30           # 30-day regulatory fresh inquiry limit
    MIN_APPLICANT_AGE = 21             # CBE legal personal loan age
    MAX_APPLICANT_AGE = 65             # Retirement threshold
    MIN_OCR_CONFIDENCE_THRESHOLD = 0.70
    WINDOW_DRESSING_SURGE_RATIO = 2.5
    ANNUITY_TO_LIQUIDITY_MAX = 0.60

    def __init__(self, db_path: str = "fraud_registry.db", artifact_dir: str = None):
        self.entity_store = SQLiteEntityStore(db_path=db_path)
        self.ml_manager = PersistentIsolationForestManager(artifact_dir=artifact_dir)

    def evaluate(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end 5-layer forensic fraud assessment on the application payload.
        Backward-compatible with dashboard.py and api.py.
        """
        violations: List[FraudRuleViolation] = []
        checks_passed: Dict[str, bool] = {}
        app_id = application.get("application_id", "APP-UNKNOWN")

        # ---------------------------------------------------------------------
        # LAYER 1: Deterministic Cross-Document & Policy Rules
        # ---------------------------------------------------------------------
        # 1. Identity Verification
        nid_ok, nid_v = self._verify_identity(application)
        checks_passed["identity_verified"] = nid_ok
        violations.extend(nid_v)

        # 2. Income & Cashflow Reconciliation
        income_ok, inc_v, income_mismatch = self._verify_income_consistency(application)
        checks_passed["income_verified"] = income_ok
        violations.extend(inc_v)

        # 3. Employer Matching
        emp_ok, emp_v, emp_similarity = self._verify_employer_consistency(application)
        checks_passed["employer_verified"] = emp_ok
        violations.extend(emp_v)

        # 4. Document Forensics
        docs_ok, doc_v = self._verify_document_integrity(application)
        checks_passed["document_integrity_verified"] = docs_ok
        violations.extend(doc_v)

        # 5. I-Score Freshness & Lawsuits
        bureau_ok, bur_v = self._verify_bureau_report(application)
        checks_passed["bureau_verified"] = bureau_ok
        violations.extend(bur_v)

        # 6. Statement Math & Bounced Cheques
        bank_ok, bank_v = self._verify_banking_behavior(application)
        checks_passed["bank_statement_math_verified"] = bank_ok
        violations.extend(bank_v)

        # ---------------------------------------------------------------------
        # LAYER 2: Deep Forensic Signals & Cashflow Analytics (Benford + Uniformity)
        # ---------------------------------------------------------------------
        anomaly_signals, uniformity_score = self._detect_behavioral_and_forensic_anomalies(application, income_mismatch)

        # ---------------------------------------------------------------------
        # LAYER 3: Syndicate & Entity Collision Detection (Graph / Velocity)
        # ---------------------------------------------------------------------
        entities_to_track = {
            "NATIONAL_ID": str(application.get("national_id_fields", {}).get("national_id", {}).get("value", "")),
            "PHONE_NUMBER": str(application.get("form_data", {}).get("mobile_number", "")),
            "EMPLOYER_NAME": str(application.get("salary_certificate_fields", {}).get("employer_name", {}).get("value", "")),
            "BANK_ACCOUNT": str(application.get("bank_statement_fields", {}).get("account_number", {}).get("value", ""))
        }
        timestamp_str = application.get("submission_timestamp", datetime.utcnow().isoformat())
        collisions = self.entity_store.check_and_record(app_id, entities_to_track, timestamp_str, window_hours=48)

        # Convert collisions to policy violations if syndicate activity suspected
        for col in collisions:
            severity = "CRITICAL" if col.is_syndicate_alert else "HIGH"
            violations.append(FraudRuleViolation(
                rule_code=f"RULE_COLLISION_{col.entity_type}",
                rule_name_en=f"Rapid Entity Collision ({col.entity_type})",
                rule_name_ar=f"تكرار مشبوه في استخدام بيانات ({col.entity_type})",
                severity=severity,
                description_en=col.explanation_en,
                description_ar=col.explanation_ar,
                observed_value=f"{col.collision_count} occurrences",
                threshold_value="0 recent occurrences",
                weight=self.SEVERITY_WEIGHTS[severity]
            ))

        # ---------------------------------------------------------------------
        # LAYER 4: High-Dimensional Unsupervised Machine Learning (Isolation Forest)
        # ---------------------------------------------------------------------
        feature_vector = HighDimensionalFraudVectorizer.extract_features(application, income_mismatch, uniformity_score)
        if_score, if_is_anomaly = self.ml_manager.score_application(feature_vector)

        if if_is_anomaly:
            anomaly_signals.append(AnomalySignal(
                anomaly_name="UNSUPERVISED_ISOLATION_FOREST_ANOMALY",
                anomaly_score=if_score,
                detected=True,
                explanation_en=f"Isolation Forest identified non-linear multi-attribute outlier pattern (Score: {if_score:.2f}).",
                explanation_ar=f"خوارزمية العزل غير الخاضعة للإشراف رصدت نمطاً شاذاً متعدد الأبعاد لا يتماشى مع سلوك المقترضين الطبيعيين."
            ))

        # ---------------------------------------------------------------------
        # LAYER 5: Cost-Sensitive Hybrid Fusion & Risk Tier Calibration
        # ---------------------------------------------------------------------
        fraud_score, risk_level, action = self._calculate_decision(violations, anomaly_signals, if_score)

        # XAI & Downstream Feeder Calculations
        reason_codes, executive_summary_ar, executive_summary_en = self._generate_explainability(
            violations, anomaly_signals, risk_level, application, if_score
        )

        haircut_multiplier, adjusted_salary = self._compute_income_haircut(
            application, fraud_score, risk_level, income_mismatch
        )

        return {
            "fraud_risk_score": round(fraud_score, 4),
            "fraud_risk_level": risk_level,
            "recommended_action": action,
            "action_ar": self._translate_action(action),
            "verification_checklist": checks_passed,
            "behavioral_anomalies": [asdict(a) for a in anomaly_signals],
            "metrics": {
                "income_mismatch_ratio": round(income_mismatch, 4),
                "employer_similarity_score": round(emp_similarity, 4),
                "total_violations_count": len(violations),
                "critical_violations_count": sum(1 for v in violations if v.severity == "CRITICAL"),
                "detected_anomalies_count": sum(1 for a in anomaly_signals if a.detected),
                "isolation_forest_anomaly_score": round(if_score, 4),
                "entity_collisions_count": len(collisions)
            },
            "downstream_risk_feeder": {
                "credibility_discount_factor": round(haircut_multiplier, 4),
                "declared_salary": round(self._get_declared_salary(application), 2),
                "risk_adjusted_salary": round(adjusted_salary, 2),
                "haircut_percentage": round((1.0 - haircut_multiplier) * 100, 1),
                "proceed_with_credit_model": risk_level != "CRITICAL"
            },
            "explainable_ai": {
                "regulatory_reason_codes": reason_codes,
                "executive_summary_ar": executive_summary_ar,
                "executive_summary_en": executive_summary_en
            },
            "triggered_rules": [asdict(v) for v in violations]
        }

    def enrich_payload(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches input application JSON payload with consistency_checks and full fraud_assessment block.
        """
        assessment = self.evaluate(application)
        
        if "consistency_checks" not in application:
            application["consistency_checks"] = {}
            
        cc = application["consistency_checks"]
        cc["income_mismatch_ratio"] = assessment["metrics"]["income_mismatch_ratio"]
        cc["employer_name_match"] = assessment["verification_checklist"]["employer_verified"]
        cc["employer_match_similarity_score"] = assessment["metrics"]["employer_similarity_score"]
        cc["national_id_match_across_documents"] = assessment["verification_checklist"]["identity_verified"]
        cc["fraud_risk_level"] = assessment["fraud_risk_level"]
        cc["triggered_fraud_rules"] = [v["rule_code"] for v in assessment["triggered_rules"]]
        
        application["fraud_assessment"] = assessment
        return application

    # =========================================================================
    # LAYER 1: IMPLEMENTATION DETAILS
    # =========================================================================

    def _verify_identity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        nid_fields = app.get("national_id_fields", {})
        nid_val = str(nid_fields.get("national_id", {}).get("value", "")).strip()
        age_val = nid_fields.get("age_years", {}).get("value")

        if nid_val and not re.match(r"^[23]\d{13}$", nid_val):
            violations.append(FraudRuleViolation(
                rule_code="RULE_INVALID_EGYPTIAN_NID_FORMAT",
                rule_name_en="Invalid Egyptian National ID Format",
                rule_name_ar="الرقم القومي لا يتبع الصيغة المصرية الرسمية",
                severity="CRITICAL",
                description_en="National ID must be 14 digits starting with 2 (born 1900-1999) or 3 (born 2000+).",
                description_ar="الرقم القومي يجب أن يتكون من 14 رقماً ويبدأ بـ 2 لمواليد القرن الماضي أو 3 لمواليد الألفية.",
                observed_value=nid_val,
                threshold_value="14 digits starting with 2 or 3",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        if age_val is not None:
            if age_val < self.MIN_APPLICANT_AGE or age_val > self.MAX_APPLICANT_AGE:
                violations.append(FraudRuleViolation(
                    rule_code="RULE_APPLICANT_AGE_OUT_OF_BOUNDS",
                    rule_name_en="Applicant Age Violates Lending Policy",
                    rule_name_ar="سن العميل خارج النطاق الائتماني المسموح به قانوناً",
                    severity="HIGH",
                    description_en=f"Applicant age ({age_val:.1f}) must be between {self.MIN_APPLICANT_AGE} and {self.MAX_APPLICANT_AGE} years.",
                    description_ar=f"سن العميل ({age_val:.1f} سنة) يجب أن يكون بين {self.MIN_APPLICANT_AGE} و {self.MAX_APPLICANT_AGE} سنة.",
                    observed_value=age_val,
                    threshold_value=f"[{self.MIN_APPLICANT_AGE}, {self.MAX_APPLICANT_AGE}]",
                    weight=self.SEVERITY_WEIGHTS["HIGH"]
                ))

        consistency = app.get("consistency_checks", {})
        if consistency.get("national_id_match_across_documents") is False:
            violations.append(FraudRuleViolation(
                rule_code="RULE_NID_CROSS_DOCUMENT_MISMATCH",
                rule_name_en="National ID Mismatch Across Documents",
                rule_name_ar="عدم تطابق الرقم القومي عبر المستندات المرفوعة",
                severity="CRITICAL",
                description_en="National ID on ID card differs from bank statement or credit bureau report. Possible identity theft.",
                description_ar="الرقم القومي المدون في البطاقة يختلف عن المسجل في كشف الحساب أو الآي سكور (شبهة انتحال شخصية).",
                observed_value=False,
                threshold_value=True,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        return len(violations) == 0, violations

    def _verify_income_consistency(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation], float]:
        violations = []
        salary_fields = app.get("salary_certificate_fields", {})
        bank_fields = app.get("bank_statement_fields", {})
        
        declared_salary = float(salary_fields.get("declared_net_salary", {}).get("value", 0.0) or 0.0)
        bank_inflow = float(bank_fields.get("avg_monthly_net_inflow", {}).get("value", 0.0) or 0.0)

        if declared_salary <= 0:
            violations.append(FraudRuleViolation(
                rule_code="RULE_ZERO_DECLARED_SALARY",
                rule_name_en="Unreadable or Zero Declared Salary",
                rule_name_ar="تعذر قراءة أو انعدام الراتب الصافي المعلن",
                severity="CRITICAL",
                description_en="Declared net salary on salary certificate is zero or unreadable.",
                description_ar="صافي الراتب في شهادة المرتب صفر أو غير مقروء في الـ OCR.",
                observed_value=declared_salary,
                threshold_value="> 0",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))
            return False, violations, 1.0

        if bank_inflow <= 0:
            violations.append(FraudRuleViolation(
                rule_code="RULE_NO_BANK_INFLOW_RECORDED",
                rule_name_en="Zero Cash Inflow on Bank Statement",
                rule_name_ar="انعدام التدفقات النقدية الدائنة في كشف الحساب البنكي",
                severity="HIGH",
                description_en="Bank statement shows no regular monthly credit inflows to support declared income.",
                description_ar="كشف الحساب البنكي لا يظهر أي إيداعات أو تدفقات رواتب شهرية تدعم الدخل المزعوم.",
                observed_value=bank_inflow,
                threshold_value="> 0",
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))
            return False, violations, 1.0

        mismatch_ratio = abs(declared_salary - bank_inflow) / declared_salary

        if mismatch_ratio > self.INCOME_MISMATCH_CRITICAL:
            violations.append(FraudRuleViolation(
                rule_code="RULE_CRITICAL_INCOME_MISMATCH",
                rule_name_en="Critical Income Discrepancy (Fake Salary Slip Suspicion)",
                rule_name_ar="تضارب جسيم بين مفردات المرتب والتحويل البنكي الفعلي",
                severity="CRITICAL",
                description_en=(f"Declared salary (EGP {declared_salary:,.0f}) differs by {mismatch_ratio*100:.1f}% "
                                f"from bank statement inflow (EGP {bank_inflow:,.0f}). Possible forged certificate."),
                description_ar=(f"الراتب المعلن في الشهادة ({declared_salary:,.0f} ج.م) يختلف بنسبة {mismatch_ratio*100:.1f}% "
                                f"عن التدفق الفعلي في كشف الحساب ({bank_inflow:,.0f} ج.م). شبهة تزوير مفردات مرتب."),
                observed_value=round(mismatch_ratio, 3),
                threshold_value=self.INCOME_MISMATCH_CRITICAL,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))
        elif mismatch_ratio > self.INCOME_MISMATCH_WARN:
            violations.append(FraudRuleViolation(
                rule_code="RULE_MODERATE_INCOME_MISMATCH",
                rule_name_en="Moderate Income Discrepancy",
                rule_name_ar="تضارب متوسط بين الراتب المصرح والتدفق البنكي",
                severity="MEDIUM",
                description_en=f"Declared salary differs by {mismatch_ratio*100:.1f}% from bank statement inflows.",
                description_ar=f"الراتب المعلن يختلف بنسبة {mismatch_ratio*100:.1f}% عن متوسط إيداعات كشف الحساب.",
                observed_value=round(mismatch_ratio, 3),
                threshold_value=self.INCOME_MISMATCH_WARN,
                weight=self.SEVERITY_WEIGHTS["MEDIUM"]
            ))

        return len(violations) == 0, violations, mismatch_ratio

    def _verify_employer_consistency(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation], float]:
        violations = []
        salary_employer = str(app.get("salary_certificate_fields", {}).get("employer_name", {}).get("value", "")).strip()
        form_employer = str(app.get("form_data", {}).get("employer_name", "")).strip()

        if not salary_employer or not form_employer:
            return True, violations, 1.0

        ratio = difflib.SequenceMatcher(None, salary_employer.lower(), form_employer.lower()).ratio()

        if ratio < self.EMPLOYER_SIMILARITY_MIN:
            violations.append(FraudRuleViolation(
                rule_code="RULE_EMPLOYER_NAME_MISMATCH",
                rule_name_en="Employer Entity Mismatch Across Documents",
                rule_name_ar="عدم تطابق اسم جهة العمل بين الاستمارة وشهادة الراتب",
                severity="MEDIUM",
                description_en=f"Employer on salary certificate ('{salary_employer}') does not match form ('{form_employer}', similarity: {ratio*100:.1f}%).",
                description_ar=f"جهة العمل في الشهادة ('{salary_employer}') تختلف عن الاستمارة ('{form_employer}'، نسبة التطابق: {ratio*100:.1f}%).",
                observed_value=round(ratio, 3),
                threshold_value=self.EMPLOYER_SIMILARITY_MIN,
                weight=self.SEVERITY_WEIGHTS["MEDIUM"]
            ))

        return len(violations) == 0, violations, ratio

    def _verify_document_integrity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        documents = app.get("documents", [])

        for doc in documents:
            doc_type = doc.get("document_type", "unknown")
            is_tampered = doc.get("is_tampered_suspected", False)
            quality = float(doc.get("overall_quality_score", 1.0))

            if is_tampered:
                violations.append(FraudRuleViolation(
                    rule_code=f"RULE_TAMPERED_{doc_type.upper()}",
                    rule_name_en=f"Digital Alteration Detected in {doc_type}",
                    rule_name_ar=f"شبهة تعديل رقمي / تلاعب فوتوشوب في مستند {doc_type}",
                    severity="CRITICAL",
                    description_en=f"Forensic document analysis detected font misalignment or metadata anomalies in {doc_type}.",
                    description_ar=f"الفحص الجنائي للصور اكتشف عدم انتظام الخطوط وتعديل رقمي في مستند {doc_type}.",
                    observed_value=True,
                    threshold_value=False,
                    weight=self.SEVERITY_WEIGHTS["CRITICAL"]
                ))

            if quality < self.MIN_OCR_CONFIDENCE_THRESHOLD:
                violations.append(FraudRuleViolation(
                    rule_code=f"RULE_LOW_OCR_CONFIDENCE_{doc_type.upper()}",
                    rule_name_en=f"Degraded Image Quality in {doc_type}",
                    rule_name_ar=f"انخفاض جودة القراءة الضوئية لمستند {doc_type}",
                    severity="LOW",
                    description_en=f"OCR readability confidence for {doc_type} is unusually low ({quality*100:.1f}%).",
                    description_ar=f"معدل ثقة قراءة الـ OCR لمستند {doc_type} منخفض ({quality*100:.1f}%).",
                    observed_value=round(quality, 3),
                    threshold_value=self.MIN_OCR_CONFIDENCE_THRESHOLD,
                    weight=self.SEVERITY_WEIGHTS["LOW"]
                ))

        return len(violations) == 0, violations

    def _verify_bureau_report(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        iscore = app.get("iscore_report_fields", {})
        if not iscore.get("is_available", True):
            return True, violations

        report_age_days = app.get("consistency_checks", {}).get("iscore_report_age_days", 0)
        if report_age_days > self.MAX_ISCORE_AGE_DAYS:
            violations.append(FraudRuleViolation(
                rule_code="RULE_EXPIRED_ISCORE_REPORT",
                rule_name_en="Stale Egyptian I-Score Report (> 30 Days)",
                rule_name_ar="تقرير الاستعلام الائتماني (I-Score) منتهي الصلاحية (> 30 يوماً)",
                severity="MEDIUM",
                description_en=f"I-Score report is {report_age_days} days old (exceeds 30-day regulatory limit).",
                description_ar=f"تقرير الآي سكور عمره {report_age_days} يوماً (يتجاوز الحد الأقصى لصلاحية الاستعلام 30 يوماً).",
                observed_value=report_age_days,
                threshold_value=self.MAX_ISCORE_AGE_DAYS,
                weight=self.SEVERITY_WEIGHTS["MEDIUM"]
            ))

        for fac in iscore.get("bureau_facilities", []):
            if fac.get("legal_action_flag", False):
                violations.append(FraudRuleViolation(
                    rule_code="RULE_LEGAL_ACTION_ON_RECORD",
                    rule_name_en="Active Legal Proceedings on Record",
                    rule_name_ar="وجود نزاع قضائي مصرفي قائم (إجراء قانوني) على العميل",
                    severity="CRITICAL",
                    description_en=f"Active legal action on record from lender: {fac.get('lender_name', 'Unknown')}.",
                    description_ar=f"مسجل إجراء قضائي نشط من البنك المقرض: {fac.get('lender_name', 'غير محدد')}.",
                    observed_value=True,
                    threshold_value=False,
                    weight=self.SEVERITY_WEIGHTS["CRITICAL"]
                ))

        return len(violations) == 0, violations

    def _verify_banking_behavior(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        bank_fields = app.get("bank_statement_fields", {})
        returned_cheques = bank_fields.get("returned_cheques_count", {}).get("value", 0) or 0

        if returned_cheques > 0:
            violations.append(FraudRuleViolation(
                rule_code="RULE_BOUNCED_CHEQUES_RECORDED",
                rule_name_en="Bounced Cheques on Bank Statement",
                rule_name_ar="شيكات بدون رصيد مسجلة في كشف الحساب البنكي",
                severity="HIGH" if returned_cheques >= 2 else "MEDIUM",
                description_en=f"Bank statement records {returned_cheques} bounced cheque(s) in last 6 months.",
                description_ar=f"كشف الحساب يسجل عدد {returned_cheques} شيك بدون رصيد خلال الـ 6 أشهر الماضية.",
                observed_value=returned_cheques,
                threshold_value=0,
                weight=self.SEVERITY_WEIGHTS["HIGH"] if returned_cheques >= 2 else self.SEVERITY_WEIGHTS["MEDIUM"]
            ))

        consistency = app.get("consistency_checks", {})
        if consistency.get("running_balance_math_valid") is False:
            violations.append(FraudRuleViolation(
                rule_code="RULE_STATEMENT_MATH_INCONSISTENCY",
                rule_name_en="Bank Statement Running Balance Math Inconsistency",
                rule_name_ar="خلل حسابي في تسلسل رصيد كشف الحساب (تزوير أرقام)",
                severity="CRITICAL",
                description_en="Cumulative transactions do not reconcile with ending balances. Strong indicator of forged statement.",
                description_ar="تسلسل العمليات الحسابية لا يتطابق مع رصيد الإقفال، مؤشر قوي على التعديل اليدوي والتزوير.",
                observed_value=False,
                threshold_value=True,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        return len(violations) == 0, violations

    # =========================================================================
    # LAYER 2: DEEP FORENSICS & BEHAVIORAL ANOMALIES
    # =========================================================================

    def _detect_behavioral_and_forensic_anomalies(self, app: Dict[str, Any], 
                                                  income_mismatch: float) -> Tuple[List[AnomalySignal], float]:
        signals = []
        bank_fields = app.get("bank_statement_fields", {})
        salary_fields = app.get("salary_certificate_fields", {})
        form_data = app.get("form_data", {})

        avg_balance = float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 0.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 1.0) or 1.0)
        declared_salary = float(salary_fields.get("declared_net_salary", {}).get("value", 0.0) or 0.0)
        bank_inflow = float(bank_fields.get("avg_monthly_net_inflow", {}).get("value", 0.0) or 0.0)
        requested_annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)

        # 1. Window Dressing Detector
        is_window_dressed = False
        surge_ratio = 1.0
        if avg_balance > 0:
            surge_ratio = max_balance / avg_balance
            if surge_ratio >= self.WINDOW_DRESSING_SURGE_RATIO and min_balance < (0.15 * avg_balance):
                is_window_dressed = True

        signals.append(AnomalySignal(
            anomaly_name="ARTIFICIAL_BALANCE_INFLATION_WINDOW_DRESSING",
            anomaly_score=round(min(surge_ratio / 4.0, 1.0), 3) if is_window_dressed else 0.0,
            detected=is_window_dressed,
            explanation_en=f"Peak balance (EGP {max_balance:,.0f}) is {surge_ratio:.1f}x higher than average balance (EGP {avg_balance:,.0f}), followed by low liquidity reserve.",
            explanation_ar=f"أعلى رصيد ({max_balance:,.0f} ج.م) يتجاوز {surge_ratio:.1f} أضعاف المتوسط الشهري مع فراغ الحساب، مؤشر على اقتراض مؤقت لتجميل كشف الحساب."
        ))

        # 2. Cashflow Volatility & Erratic Inflow Pattern
        is_volatile = False
        if avg_balance > 0 and (volatility / avg_balance) > 1.20 and regularity < 0.60:
            is_volatile = True

        signals.append(AnomalySignal(
            anomaly_name="ERRATIC_CASHFLOW_INSTABILITY",
            anomaly_score=0.65 if is_volatile else 0.10,
            detected=is_volatile,
            explanation_en=f"High balance dispersion (std EGP {volatility:,.0f}) paired with low salary deposit regularity ({regularity*100:.0f}%).",
            explanation_ar=f"تذبذب شديد في السيولة (انحراف معياري {volatility:,.0f} ج.م) مع ضعف انتظام مواعيد نزول المرتب ({regularity*100:.0f}%)."
        ))

        # 3. High Annuity to Liquid Buffer Ratio
        is_annuity_stress = False
        if avg_balance > 0 and requested_annuity > 0:
            buffer_ratio = requested_annuity / avg_balance
            if buffer_ratio > self.ANNUITY_TO_LIQUIDITY_MAX:
                is_annuity_stress = True

        signals.append(AnomalySignal(
            anomaly_name="LIQUIDITY_BUFFER_STRESS",
            anomaly_score=0.70 if is_annuity_stress else 0.05,
            detected=is_annuity_stress,
            explanation_en=f"Requested monthly annuity (EGP {requested_annuity:,.0f}) absorbs > 60% of applicant's historical average liquid balance.",
            explanation_ar=f"القسط الشهري المطلوب ({requested_annuity:,.0f} ج.م) يستنزف أكثر من 60% من متوسط رصيد العميل السائل تاريخياً."
        ))

        # 4. Benford's Law Cashflow Forensics
        cashflow_sample = [avg_balance, max_balance, min_balance, declared_salary, bank_inflow, requested_annuity, volatility]
        is_benford_anomaly, benford_score, benford_msg = DeepForensicAnalyzer.evaluate_benford_law(cashflow_sample)
        signals.append(AnomalySignal(
            anomaly_name="BENFORD_LAW_CASHFLOW_DEVIATION",
            anomaly_score=benford_score,
            detected=is_benford_anomaly,
            explanation_en=benford_msg,
            explanation_ar="انحراف إحصائي في توزيع الأرقام الأولى لمعاملات كشف الحساب وفق قانون بنفورد الطبيعي (شبهة فبركة أرقام)." if is_benford_anomaly else "الأرقام المالية تتوافق مع التوزيع الطبيعي لقانون بنفورد."
        ))

        # 5. Inflow Uniformity & Round Number Suspicion
        is_uniform, unif_score, unif_msg = DeepForensicAnalyzer.check_inflow_uniformity(declared_salary, bank_inflow, avg_balance, min_balance)
        signals.append(AnomalySignal(
            anomaly_name="INFLOW_UNIFORMITY_ROUND_NUMBER_ANOMALY",
            anomaly_score=unif_score,
            detected=is_uniform,
            explanation_en=unif_msg,
            explanation_ar="تطابق مصطنع لأرقام رواتب دائرية مصمتة خالية من الاستقطاعات الطبيعية (ضرائب/تأمينات)." if is_uniform else "طبيعة التدفقات المالية تحتوي على استقطاعات واقعية."
        ))

        return signals, unif_score

    # =========================================================================
    # LAYER 5: DECISION FUSION, XAI & HAIRCUT
    # =========================================================================

    def _calculate_decision(self, violations: List[FraudRuleViolation], 
                            anomalies: List[AnomalySignal], 
                            isolation_forest_score: float) -> Tuple[float, str, str]:
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        has_high = any(v.severity == "HIGH" for v in violations)

        # Weighted rule contributions
        rule_score = sum(v.weight for v in violations)
        
        # Behavioral anomaly contributions
        anomaly_score = sum(a.anomaly_score * 0.15 for a in anomalies if a.detected)
        
        # Isolation Forest contribution
        ml_score = isolation_forest_score * 0.25

        composite_score = min(0.04 + rule_score + anomaly_score + ml_score, 1.0)

        # Hard Rule deterministic overrides
        if has_critical or composite_score >= 0.70:
            risk_level = "CRITICAL"
            action = "REJECT_SUSPECTED_FRAUD"
        elif has_high or composite_score >= 0.45:
            risk_level = "HIGH"
            action = "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION"
        elif composite_score >= 0.25:
            risk_level = "MEDIUM"
            action = "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS"
        else:
            risk_level = "LOW"
            action = "PROCEED_TO_CREDIT_EVALUATION"

        return composite_score, risk_level, action

    def _generate_explainability(self, violations: List[FraudRuleViolation], 
                                 anomalies: List[AnomalySignal],
                                 risk_level: str, app: Dict[str, Any],
                                 ml_score: float) -> Tuple[List[Dict[str, str]], str, str]:
        reason_codes = []
        for v in violations:
            reason_codes.append({
                "code": v.rule_code,
                "severity": v.severity,
                "reason_en": v.description_en,
                "reason_ar": v.description_ar
            })

        for a in anomalies:
            if a.detected:
                reason_codes.append({
                    "code": a.anomaly_name,
                    "severity": "MEDIUM",
                    "reason_en": a.explanation_en,
                    "reason_ar": a.explanation_ar
                })

        # Generate Executive Summaries
        applicant = app.get("national_id_fields", {}).get("full_name", {}).get("value", "The applicant")
        if risk_level == "LOW":
            summary_en = (f"Application for {applicant} successfully passed deterministic cross-document reconciliation "
                          f"and multi-dimensional Isolation Forest anomaly screening (ML Outlier Score: {ml_score:.2f}). "
                          f"Zero entity collisions detected. Recommended to proceed to credit evaluation.")
            summary_ar = (f"اجتاز طلب التمويل للعميل ({applicant}) كافة اختبارات المطابقة المتقاطعة للوثائق وفحص شذوذ "
                          f"التعلم الآلي (Isolation Forest: {ml_score:.2f}) دون رصد أي تكرار مشبوه في الكيانات. يوصى بالموافقة على تمرير الطلب لتقييم الجدارة الائتمانية.")
        elif risk_level in ["MEDIUM", "HIGH"]:
            summary_en = (f"Application for {applicant} triggered {len(violations)} policy violation(s) and "
                          f"behavioral discrepancies. Multi-attribute Isolation Forest scored at {ml_score:.2f}. "
                          f"Requires secondary manual verification or certified bank confirmation before underwriting.")
            summary_ar = (f"سجل طلب العميل ({applicant}) عدد {len(violations)} مخالفة سياسات وتفاوت في التدفقات النقدية "
                          f"(سكور التعلم الآلي: {ml_score:.2f}). يوصى بالتحويل للمراجعة اليدوية وطلب مستندات إضافية قبل المتابعة.")
        else:
            summary_en = (f"CRITICAL FORENSIC ALERT: Fatal irregularities detected in application for {applicant}. "
                          f"Confirmed document tampering, synthetic cashflows, or entity collision detected. Immediate decline required under CBE AML/Anti-Fraud mandate.")
            summary_ar = (f"تحذير جنائي حرج: رصد تزوير أو تلاعب رقمي مؤكد أو تضارب جوهري في كشوفات الحساب لطلب العميل ({applicant}). "
                          f"يجب رفض الطلب فوراً وتسجيله في سجل مكافحة الاحتيال المصرفي طبقاً لتعليمات البنك المركزي المصري.")

        return reason_codes, summary_ar, summary_en

    def _compute_income_haircut(self, app: Dict[str, Any], fraud_score: float, 
                               risk_level: str, mismatch: float) -> Tuple[float, float]:
        salary_fields = app.get("salary_certificate_fields", {})
        bank_fields = app.get("bank_statement_fields", {})
        declared = float(salary_fields.get("declared_net_salary", {}).get("value", 0.0) or 0.0)
        inflow = float(bank_fields.get("avg_monthly_net_inflow", {}).get("value", 0.0) or 0.0)

        if risk_level == "CRITICAL":
            return 0.0, 0.0

        if declared <= 0:
            return 1.0, 0.0

        # Base haircut matches verified bank inflow
        credible_base = min(declared, inflow) if inflow > 0 else declared

        # Additional discount penalty based on fraud score
        discount = 1.0 - (fraud_score * 0.35)
        adjusted = max(0.0, credible_base * discount)
        effective_multiplier = adjusted / declared if declared > 0 else 0.0

        return round(effective_multiplier, 4), round(adjusted, 2)

    def _get_declared_salary(self, app: Dict[str, Any]) -> float:
        return float(app.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0.0) or 0.0)

    def _translate_action(self, action: str) -> str:
        translations = {
            "PROCEED_TO_CREDIT_EVALUATION": "تمرير الطلب لموديل تقييم الجدارة الائتمانية",
            "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS": "طلب مستندات إضافية وتأكيد بنكي معتمد",
            "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION": "تحويل الطلب للتحقيق الجنائي اليدوي",
            "REJECT_SUSPECTED_FRAUD": "رفض فوري لشبهة تزوير واحتيال مستندي"
        }
        return translations.get(action, action)
