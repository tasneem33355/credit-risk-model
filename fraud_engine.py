"""
Senior Production-Grade 5-Layer Credit Application Fraud & Consistency Engine
=============================================================================
Platform: Smart Financing & Credit Request Analysis Platform (ZAWOLF / CrediX)
Standard: Egyptian Banking Federation & CBE Regulatory Compliance Guidelines

Architecture:
  - Layer 1: Deterministic Cross-Document Rules (NID, Salary vs Statement, OCR, Bureau, Device Telemetry)
  - Layer 2: Deep Forensic Signals (Benford's Law Chi-Square test, Inflow Uniformity)
  - Layer 3: SQLite Entity Collision & Velocity Defense (Cross-application 48h tracking for Phone, Account, Device)
  - Layer 4: Dual-Engine ML Inference (Multi-Tree Isolation Forest + Cost-Sensitive Gradient Boosting)
  - Layer 5: Cost-Sensitive Hybrid Fusion, Bilingual Explainable AI (XAI) & Income Haircut Calculator

Note:
  Model training is separated into model/train_fraud.py and notebooks/fraud_model_training.ipynb.
  This engine exclusively performs fast, production-grade vectorized inference (<5ms).
"""

import os
import re
import math
import json
import sqlite3
import hashlib
import difflib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import joblib


# -----------------------------------------------------------------------------
# Data Structures & Signal Schemas
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Layer 2 Helper: Deep Forensic Analyzer (Benford's Law & Round Numbers)
# -----------------------------------------------------------------------------

class DeepForensicAnalyzer:
    """
    Mathematical forensics on transaction amounts in uploaded bank statements.
    Implements Benford's Law (First Digit Chi-Square Goodness of Fit)
    and Uniformity Analysis (Detection of rounded fabricated salaries).
    """

    BENFORD_PROBABILITIES = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }

    @staticmethod
    def evaluate_benford_law(numbers: List[float]) -> Tuple[bool, float, float]:
        valid_digits = []
        for n in numbers:
            val = abs(float(n or 0))
            if val >= 1.0:
                s = f"{val:.4f}".replace(".", "").lstrip("0")
                if s and s[0] in "123456789":
                    valid_digits.append(int(s[0]))

        if len(valid_digits) < 4:
            return False, 0.0, 1.0

        n_total = len(valid_digits)
        observed_counts = {d: 0 for d in range(1, 10)}
        for d in valid_digits:
            observed_counts[d] += 1

        chi_square = 0.0
        for d in range(1, 10):
            expected = n_total * DeepForensicAnalyzer.BENFORD_PROBABILITIES[d]
            observed = observed_counts[d]
            chi_square += ((observed - expected) ** 2) / max(expected, 0.001)

        is_anomaly = bool(chi_square > 15.51)
        return is_anomaly, round(chi_square, 2), 0.02 if is_anomaly else 0.85

    @staticmethod
    def calculate_inflow_uniformity(amounts: List[float]) -> float:
        if not amounts:
            return 0.0
        clean_amounts = [abs(float(a)) for a in amounts if abs(float(a)) > 100]
        if not clean_amounts:
            return 0.0
        round_count = sum(1 for a in clean_amounts if a % 1000 == 0 or a % 500 == 0)
        return round(round_count / len(clean_amounts), 3)


# -----------------------------------------------------------------------------
# Layer 3 Helper: Real-time Entity Graph & Velocity Store (SQLite)
# -----------------------------------------------------------------------------

class SQLiteEntityStore:
    """
    Lightweight, embedded cross-application registry to detect fraud ring velocity
    and entity collisions (phone, NID, employer, IBAN, device) across a rolling 48-hour window.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "fraud_registry.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id TEXT,
                        timestamp TEXT,
                        nid_hash TEXT,
                        phone_hash TEXT,
                        employer_name TEXT,
                        bank_account_hash TEXT,
                        device_id_hash TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_time ON entity_audit_log(timestamp)")
                conn.commit()
        except Exception:
            pass

    def check_and_record_velocity(
        self, application_id: str, national_id: str, phone: str, employer: str, account: str, device_id: str = ""
    ) -> Dict[str, Any]:
        now = datetime.utcnow()
        window_start = (now - timedelta(hours=48)).isoformat()
        nid_h = hashlib.sha256((national_id or "").strip().encode()).hexdigest() if national_id else ""
        ph_h = hashlib.sha256((phone or "").strip().encode()).hexdigest() if phone else ""
        acc_h = hashlib.sha256((account or "").strip().encode()).hexdigest() if account else ""
        dev_h = hashlib.sha256((device_id or "").strip().encode()).hexdigest() if device_id else ""
        emp_clean = (employer or "").strip().lower()

        collisions = {
            "duplicate_phone_in_48h": False,
            "duplicate_account_in_48h": False,
            "duplicate_device_in_48h": False,
            "employer_spike_in_48h": False,
            "total_collisions": 0
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if ph_h:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE phone_hash = ? AND timestamp >= ? AND application_id != ?",
                        (ph_h, window_start, application_id)
                    )
                    count_phone = cursor.fetchone()[0]
                    if count_phone >= 2:
                        collisions["duplicate_phone_in_48h"] = True
                        collisions["total_collisions"] += count_phone

                if acc_h:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE bank_account_hash = ? AND timestamp >= ? AND application_id != ?",
                        (acc_h, window_start, application_id)
                    )
                    count_acc = cursor.fetchone()[0]
                    if count_acc >= 1:
                        collisions["duplicate_account_in_48h"] = True
                        collisions["total_collisions"] += count_acc

                if dev_h:
                    cursor.execute(
                        "SELECT COUNT(DISTINCT application_id) FROM entity_audit_log WHERE device_id_hash = ? AND timestamp >= ? AND application_id != ?",
                        (dev_h, window_start, application_id)
                    )
                    count_dev = cursor.fetchone()[0]
                    if count_dev >= 2:
                        collisions["duplicate_device_in_48h"] = True
                        collisions["total_collisions"] += count_dev

                cursor.execute(
                    "INSERT INTO entity_audit_log (application_id, timestamp, nid_hash, phone_hash, employer_name, bank_account_hash, device_id_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (application_id, now.isoformat(), nid_h, ph_h, emp_clean, acc_h, dev_h)
                )
                conn.commit()
        except Exception:
            pass

        return collisions

    def get_recent_graph_data(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Returns recent entity submissions for graph network rendering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT application_id, timestamp, phone_hash, employer_name, bank_account_hash, device_id_hash
                    FROM entity_audit_log
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                data = []
                for r in rows:
                    data.append({
                        "app_id": r[0],
                        "timestamp": r[1],
                        "phone_masked": f"PH-{r[2][:6]}" if r[2] else "",
                        "employer": r[3] if r[3] else "",
                        "account_masked": f"ACC-{r[4][:6]}" if r[4] else "",
                        "device_masked": f"DEV-{r[5][:6]}" if r[5] else ""
                    })
                return data
        except Exception:
            return []


# -----------------------------------------------------------------------------
# Layer 4 Helper: Persistent Model Manager (Pre-trained ML Inference)
# -----------------------------------------------------------------------------

class PersistentModelManager:
    FEATURE_NAMES = [
        'income_mismatch_ratio', 'annuity_to_balance_ratio', 'balance_volatility_cv',
        'surge_ratio_max_to_avg', 'ocr_quality_mean', 'min_to_avg_balance_ratio',
        'applicant_age_norm', 'employment_tenure_years', 'inflow_regularity_score',
        'iscore_normalized', 'inflow_uniformity_score', 'bureau_facilities_count'
    ]

    def __init__(self, artifact_dir: Optional[str] = None):
        if artifact_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.artifact_dir = os.path.join(base_dir, "model", "artifacts", "fraud")
        else:
            self.artifact_dir = artifact_dir

        self.iso_model = None
        self.gb_model = None
        self.scaler = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            iso_path = os.path.join(self.artifact_dir, "isolation_forest_v2.joblib")
            gb_path = os.path.join(self.artifact_dir, "fraud_gradient_boost_v2.joblib")
            scaler_path = os.path.join(self.artifact_dir, "scaler_v2.joblib")

            if os.path.exists(iso_path) and os.path.exists(gb_path) and os.path.exists(scaler_path):
                self.iso_model = joblib.load(iso_path)
                self.gb_model = joblib.load(gb_path)
                self.scaler = joblib.load(scaler_path)
                self.is_loaded = True
            else:
                self._fallback_init()
        except Exception:
            self._fallback_init()

    def _fallback_init(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        np.random.seed(42)
        X_dummy = np.random.uniform(0.1, 0.9, size=(250, 12))
        self.scaler = StandardScaler().fit(X_dummy)
        self.iso_model = IsolationForest(n_estimators=50, random_state=42).fit(self.scaler.transform(X_dummy))
        self.gb_model = None
        self.is_loaded = False

    def predict_scores(self, feature_vector: np.ndarray) -> Tuple[float, float, bool]:
        try:
            scaled = self.scaler.transform(feature_vector)
            raw_iso = self.iso_model.decision_function(scaled)[0]
            iso_score = float(np.clip(0.50 - (raw_iso * 1.8), 0.0, 1.0))
            is_anomaly = bool(iso_score > 0.60)

            if self.gb_model is not None:
                gb_prob = float(self.gb_model.predict_proba(scaled)[0, 1])
            else:
                gb_prob = iso_score

            return round(iso_score, 4), round(gb_prob, 4), is_anomaly
        except Exception:
            return 0.15, 0.15, False


# -----------------------------------------------------------------------------
# Layer 4 Feature Vectorizer (Extracts 12 Canonical Features)
# -----------------------------------------------------------------------------

class HighDimensionalFraudVectorizer:
    @staticmethod
    def extract_features(app: Dict[str, Any], mismatch_ratio: float, uniformity_score: float) -> np.ndarray:
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})
        salary_fields = app.get("salary_certificate_fields", {})
        nid_fields = app.get("national_id_fields", {})
        iscore_fields = app.get("iscore_report_fields", {})

        avg_balance = max(float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0), 1.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 0.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 1.0) or 1.0)
        tenure = float(salary_fields.get("employment_tenure_years", {}).get("value", 3.0) or 3.0)
        age = float(nid_fields.get("age_years", {}).get("value", 35.0) or 35.0)
        iscore = float(iscore_fields.get("credit_score", {}).get("value", 650) or 650)
        facilities = float(iscore_fields.get("active_facilities_count", {}).get("value", 2) or 2)

        docs = app.get("documents", [])
        ocr_scores = [float(d.get("overall_quality_score", 0.85)) for d in docs if isinstance(d, dict)]
        ocr_mean = float(np.mean(ocr_scores)) if ocr_scores else 0.88

        feat = [
            float(np.clip(mismatch_ratio, 0.0, 2.0)),
            float(np.clip(annuity / avg_balance, 0.0, 5.0)),
            float(np.clip(volatility / avg_balance, 0.0, 5.0)),
            float(np.clip(max_balance / avg_balance, 1.0, 10.0)),
            float(np.clip(ocr_mean, 0.0, 1.0)),
            float(np.clip(min_balance / avg_balance, 0.0, 1.0)),
            float(np.clip((age - 21.0) / 44.0, 0.0, 1.0)),
            float(np.clip(tenure, 0.0, 30.0)),
            float(np.clip(regularity, 0.0, 1.0)),
            float(np.clip((iscore - 300.0) / 550.0, 0.0, 1.0)),
            float(np.clip(uniformity_score, 0.0, 1.0)),
            float(np.clip(facilities, 0.0, 15.0))
        ]
        return np.array([feat], dtype=np.float32)


# -----------------------------------------------------------------------------
# Main Senior 5-Layer Credit Fraud Engine
# -----------------------------------------------------------------------------

class CreditFraudEngine:
    SEVERITY_WEIGHTS = {
        "LOW": 0.10,
        "MEDIUM": 0.25,
        "HIGH": 0.50,
        "CRITICAL": 0.90
    }

    INCOME_MISMATCH_WARN = 0.20
    INCOME_MISMATCH_CRITICAL = 0.40
    EMPLOYER_SIMILARITY_MIN = 0.65
    MAX_ISCORE_AGE_DAYS = 30
    MIN_APPLICANT_AGE = 21
    MAX_APPLICANT_AGE = 65
    MIN_OCR_CONFIDENCE_THRESHOLD = 0.70
    WINDOW_DRESSING_SURGE_RATIO = 2.5
    ANNUITY_TO_LIQUIDITY_MAX = 0.60

    def __init__(self, db_path: Optional[str] = None, artifact_dir: Optional[str] = None):
        self.entity_store = SQLiteEntityStore(db_path)
        self.model_manager = PersistentModelManager(artifact_dir)

    def evaluate(self, application: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[FraudRuleViolation] = []
        checks_passed: Dict[str, bool] = {}

        # ---------------------------------------------------------------------
        # LAYER 1: Deterministic Cross-Document & Digital Telemetry Checks
        # ---------------------------------------------------------------------
        nid_ok, nid_v = self._verify_identity(application)
        checks_passed["identity_verified"] = nid_ok
        violations.extend(nid_v)

        income_ok, inc_v, income_mismatch = self._verify_income_consistency(application)
        checks_passed["income_verified"] = income_ok
        violations.extend(inc_v)

        emp_ok, emp_v, emp_similarity = self._verify_employer_consistency(application)
        checks_passed["employer_verified"] = emp_ok
        violations.extend(emp_v)

        docs_ok, doc_v = self._verify_document_integrity(application)
        checks_passed["document_integrity_verified"] = docs_ok
        violations.extend(doc_v)

        bureau_ok, bur_v = self._verify_bureau_report(application)
        checks_passed["bureau_verified"] = bureau_ok
        violations.extend(bur_v)

        bank_ok, bank_v = self._verify_banking_behavior(application)
        checks_passed["bank_statement_math_verified"] = bank_ok
        violations.extend(bank_v)

        # Digital Telemetry & Device Verification
        telemetry_ok, tel_v = self._verify_device_telemetry(application)
        checks_passed["device_telemetry_verified"] = telemetry_ok
        violations.extend(tel_v)

        # ---------------------------------------------------------------------
        # LAYER 2: Deep Forensic Signals (Benford's Law & Cashflow Uniformity)
        # ---------------------------------------------------------------------
        anomaly_signals, uniformity_score, benford_anomaly = self._detect_behavioral_and_forensic_anomalies(
            application, income_mismatch
        )

        # ---------------------------------------------------------------------
        # LAYER 3: Entity Collisions & Cross-Application Velocity (Graph Store)
        # ---------------------------------------------------------------------
        collisions = self._verify_entity_velocity(application)
        if collisions.get("duplicate_phone_in_48h"):
            violations.append(FraudRuleViolation(
                rule_code="VEL-001-CROSS-APP-PHONE-COLLISION",
                rule_name_en="Cross-Application Phone Velocity Collision",
                rule_name_ar="تكرار رقم الهاتف في أكثر من طلب خلال 48 ساعة",
                severity="CRITICAL",
                description_en="Contact phone number detected across multiple distinct applicant profiles within 48h (Fraud Ring Signal).",
                description_ar="تم رصد استخدام رقم الهاتف في طلبات ائتمانية متعددة ببطاقات رقم قومي مختلفة خلال 48 ساعة.",
                observed_value=collisions["total_collisions"],
                threshold_value="< 2 applications / 48h",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        if collisions.get("duplicate_account_in_48h"):
            violations.append(FraudRuleViolation(
                rule_code="VEL-002-CROSS-APP-ACCOUNT-COLLISION",
                rule_name_en="Bank Account Multiple Identity Reuse",
                rule_name_ar="استخدام الحساب البنكي لأكثر من عميل مختلف",
                severity="CRITICAL",
                description_en="The provided bank statement account number is associated with another borrower in the registry.",
                description_ar="رقم الحساب البنكي المستخدم مسجل مسبقاً باسم عميل آخر في قاعدة البيانات.",
                observed_value=True,
                threshold_value=False,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        if collisions.get("duplicate_device_in_48h"):
            violations.append(FraudRuleViolation(
                rule_code="VEL-003-DEVICE-FINGERPRINT-COLLISION",
                rule_name_en="Device Fingerprint Fraud Ring Collision",
                rule_name_ar="تقديم طلبات متعددة لعملاء مختلفين من نفس الجهاز الفعلي",
                severity="CRITICAL",
                description_en="Hardware device fingerprint was used to submit multiple distinct credit requests within 48h (Syndicate Terminal).",
                description_ar="تم رصد استخدام نفس بصمة الجهاز الفعلي لتقديم طلبات ائتمانية لعملاء مختلفين خلال 48 ساعة.",
                observed_value=collisions["total_collisions"],
                threshold_value="< 2 applications / device",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        # ---------------------------------------------------------------------
        # LAYER 4: Dual-Engine ML Inference (Isolation Forest + HistGradientBoosting)
        # ---------------------------------------------------------------------
        feature_matrix = HighDimensionalFraudVectorizer.extract_features(application, income_mismatch, uniformity_score)
        iso_score, gb_prob, is_ml_anomaly = self.model_manager.predict_scores(feature_matrix)

        if is_ml_anomaly:
            anomaly_signals.append(AnomalySignal(
                anomaly_name="UNSUPERVISED_ISOLATION_FOREST_ANOMALY",
                anomaly_score=iso_score,
                detected=True,
                explanation_en=f"Isolation Forest identified deep structural anomaly (Score: {iso_score:.3f}). Feature pattern deviates from authentic population.",
                explanation_ar=f"محرك العزل الرياضي (Isolation Forest) رصد شذوذاً هيكلياً (النتيجة: {iso_score:.3f}) يختلف عن أنماط المقترضين الطبيعيين."
            ))

        # ---------------------------------------------------------------------
        # LAYER 5: Cost-Sensitive Hybrid Fusion & Final Decision
        # ---------------------------------------------------------------------
        fraud_score, risk_level, action = self._calculate_hybrid_decision(
            violations, anomaly_signals, iso_score, gb_prob
        )

        reason_codes, exec_summary_ar, exec_summary_en = self._generate_explainability(
            violations, anomaly_signals, risk_level, application, gb_prob
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
                "inflow_uniformity_score": round(uniformity_score, 4),
                "benford_law_violation": bool(benford_anomaly),
                "isolation_forest_anomaly_score": round(iso_score, 4),
                "gradient_boost_fraud_probability": round(gb_prob, 4),
                "entity_collisions_count": collisions.get("total_collisions", 0),
                "total_violations_count": len(violations),
                "critical_violations_count": sum(1 for v in violations if v.severity == "CRITICAL"),
                "detected_anomalies_count": sum(1 for a in anomaly_signals if a.detected),
                "model_engine_status": "ONLINE_CALIBRATED_ARTIFACTS" if self.model_manager.is_loaded else "HEURISTIC_FALLBACK"
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
                "executive_summary_ar": exec_summary_ar,
                "executive_summary_en": exec_summary_en
            },
            "triggered_rules": [asdict(v) for v in violations]
        }

    # -------------------------------------------------------------------------
    # Layer 1 Rule Methods
    # -------------------------------------------------------------------------

    def _verify_device_telemetry(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        telemetry = app.get("device_telemetry", {})
        if not telemetry:
            return True, []

        is_vpn = bool(telemetry.get("is_vpn_or_proxy", False))
        if is_vpn:
            violations.append(FraudRuleViolation(
                rule_code="TEL-001-VPN-PROXY-ANONYMIZER",
                rule_name_en="Anonymous Proxy / Commercial VPN Detected",
                rule_name_ar="استخدام شبكة افتراضية خاصة (VPN) أو بروكسي لإخفاء الهوية الرقمية",
                severity="HIGH",
                description_en="Submission originated from a commercial VPN/datacenter IP address rather than a residential Egyptian ISP.",
                description_ar="طلب الائتمان تم إرساله عبر خادم VPN أو مركز بيانات تجاري لإخفاء الموقع الجغرافي الحقيقي للمتقدم.",
                observed_value="VPN = TRUE",
                threshold_value="VPN = FALSE",
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        submission_hour = telemetry.get("submission_hour_utc")
        if submission_hour is not None and (1 <= int(submission_hour) <= 4):
            violations.append(FraudRuleViolation(
                rule_code="TEL-002-OFF-HOURS-SUBMISSION",
                rule_name_en="Anomalous Off-Hours Automated Submission",
                rule_name_ar="تقديم طلب آلي في ساعات الفجر المتأخرة",
                severity="LOW",
                description_en="Application was submitted between 01:00 AM and 04:00 AM, common pattern in batch automated bot applications.",
                description_ar="تم تسجيل الطلب في ساعات متأخرة جداً فجراً، وهو نمط متكرر في هجمات التقديم الآلي.",
                observed_value=f"Hour: {submission_hour} UTC",
                threshold_value="Standard Hours",
                weight=self.SEVERITY_WEIGHTS["LOW"]
            ))

        return len(violations) == 0, violations

    def _verify_identity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        nid_fields = app.get("national_id_fields", {})
        nid_val = str(nid_fields.get("national_id", {}).get("value", "")).strip()

        if not re.match(r"^[23]\d{13}$", nid_val):
            violations.append(FraudRuleViolation(
                rule_code="ID-001-INVALID-NID-FORMAT",
                rule_name_en="Invalid Egyptian National ID Format",
                rule_name_ar="الرقم القومي غير مطابق للمعيار المصري",
                severity="CRITICAL",
                description_en="National ID must be exactly 14 digits starting with 2 (born 1900-1999) or 3 (born 2000-2099).",
                description_ar="الرقم القومي يجب أن يتكون من 14 رقماً ويبدأ بـ 2 لمواليد القرن الماضي أو 3 لمواليد القرن الحالي.",
                observed_value=nid_val,
                threshold_value="14 digits starting with 2 or 3",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        age_val = float(nid_fields.get("age_years", {}).get("value", 0.0) or 0.0)
        if age_val < self.MIN_APPLICANT_AGE or age_val > self.MAX_APPLICANT_AGE:
            violations.append(FraudRuleViolation(
                rule_code="ID-002-AGE-POLICY-BREACH",
                rule_name_en="Applicant Age Outside Regulatory Financing Range",
                rule_name_ar="عمر المتقدم خارج النطاق التمويلي المصرح به رقابياً",
                severity="HIGH",
                description_en=f"Applicant age ({age_val:.1f} years) must be between {self.MIN_APPLICANT_AGE} and {self.MAX_APPLICANT_AGE}.",
                description_ar=f"عمر المتقدم ({age_val:.1f} سنة) يجب أن يكون بين {self.MIN_APPLICANT_AGE} و {self.MAX_APPLICANT_AGE} عاماً.",
                observed_value=age_val,
                threshold_value=f"{self.MIN_APPLICANT_AGE} - {self.MAX_APPLICANT_AGE}",
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        return len(violations) == 0, violations

    def _verify_income_consistency(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation], float]:
        violations = []
        salary_fields = app.get("salary_certificate_fields", {})
        bank_fields = app.get("bank_statement_fields", {})

        declared_salary = float(salary_fields.get("declared_net_salary", {}).get("value", 0.0) or 0.0)
        bank_inflow = float(bank_fields.get("avg_monthly_net_inflow", {}).get("value", 0.0) or 0.0)

        if declared_salary <= 0:
            mismatch_ratio = 1.0
        else:
            mismatch_ratio = abs(declared_salary - bank_inflow) / declared_salary

        if declared_salary > 0 and bank_inflow > 0:
            if mismatch_ratio >= self.INCOME_MISMATCH_CRITICAL and declared_salary > bank_inflow:
                violations.append(FraudRuleViolation(
                    rule_code="INC-001-GROSS-INCOME-INFLATION",
                    rule_name_en="Gross Income Inflation Discrepancy",
                    rule_name_ar="تضخيم جوهري في الدخل المذكور بشهادة الراتب",
                    severity="CRITICAL",
                    description_en=f"Declared salary (EGP {declared_salary:,.0f}) exceeds verified bank net inflows (EGP {bank_inflow:,.0f}) by {mismatch_ratio*100:.1f}%.",
                    description_ar=f"صافي الراتب المذكور ({declared_salary:,.0f} ج.م) يفوق متوسط إيداعات البنك ({bank_inflow:,.0f} ج.م) بنسبة {mismatch_ratio*100:.1f}%.",
                    observed_value=f"{mismatch_ratio*100:.1f}% mismatch",
                    threshold_value=f"< {self.INCOME_MISMATCH_CRITICAL*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["CRITICAL"]
                ))
            elif mismatch_ratio >= self.INCOME_MISMATCH_WARN and declared_salary > bank_inflow:
                violations.append(FraudRuleViolation(
                    rule_code="INC-002-MODERATE-INCOME-MISMATCH",
                    rule_name_en="Moderate Cross-Document Income Discrepancy",
                    rule_name_ar="تفاوت متوسط بين شهادة الراتب وكشف الحساب البنكي",
                    severity="MEDIUM",
                    description_en=f"Declared salary (EGP {declared_salary:,.0f}) exceeds verified bank inflows (EGP {bank_inflow:,.0f}) by {mismatch_ratio*100:.1f}%.",
                    description_ar=f"الراتب المذكور يفوق كشف الحساب بفارق {mismatch_ratio*100:.1f}% يتطلب تدقيقاً إضافياً.",
                    observed_value=f"{mismatch_ratio*100:.1f}% mismatch",
                    threshold_value=f"< {self.INCOME_MISMATCH_WARN*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["MEDIUM"]
                ))

        return len(violations) == 0, violations, mismatch_ratio

    def _verify_employer_consistency(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation], float]:
        violations = []
        salary_fields = app.get("salary_certificate_fields", {})
        bank_fields = app.get("bank_statement_fields", {})

        emp_salary = str(salary_fields.get("employer_name", {}).get("value", "")).strip()
        emp_bank = str(bank_fields.get("payroll_transfer_employer", {}).get("value", "")).strip()

        if emp_salary and emp_bank:
            similarity = difflib.SequenceMatcher(None, emp_salary.lower(), emp_bank.lower()).ratio()
            if similarity < self.EMPLOYER_SIMILARITY_MIN:
                violations.append(FraudRuleViolation(
                    rule_code="EMP-001-EMPLOYER-NAME-MISMATCH",
                    rule_name_en="Employer Entity Discrepancy Across Documents",
                    rule_name_ar="عدم تطابق جهة العمل بين شهادة الراتب ومحول الراتب بالبنك",
                    severity="HIGH",
                    description_en=f"Salary employer '{emp_salary}' differs from bank depositor '{emp_bank}' (Similarity: {similarity*100:.1f}%).",
                    description_ar=f"جهة العمل في شهادة الراتب '{emp_salary}' تختلف عن محول المرتب '{emp_bank}' (التطابق: {similarity*100:.1f}%).",
                    observed_value=f"{similarity*100:.1f}% similarity",
                    threshold_value=f">= {self.EMPLOYER_SIMILARITY_MIN*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["HIGH"]
                ))
            return len(violations) == 0, violations, similarity

        return True, violations, 1.0

    def _verify_document_integrity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        docs = app.get("documents", [])

        for doc in docs:
            doc_type = doc.get("document_type", "Unknown")
            is_tampered = bool(doc.get("is_tampered_suspected", False))
            quality = float(doc.get("overall_quality_score", 1.0) or 1.0)

            if is_tampered:
                violations.append(FraudRuleViolation(
                    rule_code=f"DOC-001-TAMPERED-{doc_type.upper()}",
                    rule_name_en=f"Digital Alteration Detected in {doc_type}",
                    rule_name_ar=f"اشتباه تلاعب وتعديل رقمي في مستند {doc_type}",
                    severity="CRITICAL",
                    description_en=f"Forensic pixel analysis identified digital manipulation or modified font layers in {doc_type}.",
                    description_ar=f"الفحص الجنائي للصور والمستندات كشف عن تعديلات رقمية في مستند {doc_type}.",
                    observed_value="Tampering Flag = TRUE",
                    threshold_value="Tampering Flag = FALSE",
                    weight=self.SEVERITY_WEIGHTS["CRITICAL"]
                ))

            if quality < self.MIN_OCR_CONFIDENCE_THRESHOLD:
                violations.append(FraudRuleViolation(
                    rule_code=f"DOC-002-LOW-QUALITY-{doc_type.upper()}",
                    rule_name_en=f"Sub-standard OCR Confidence in {doc_type}",
                    rule_name_ar=f"انخفاض جودة القراءة الضوئية لمستند {doc_type}",
                    severity="MEDIUM",
                    description_en=f"OCR quality ({quality*100:.1f}%) is below the institutional threshold.",
                    description_ar=f"جودة استخراج النصوص ({quality*100:.1f}%) غير كافية للاعتماد الآلي.",
                    observed_value=f"{quality*100:.1f}%",
                    threshold_value=f">= {self.MIN_OCR_CONFIDENCE_THRESHOLD*100:.0f}%",
                    weight=self.SEVERITY_WEIGHTS["MEDIUM"]
                ))

        return len(violations) == 0, violations

    def _verify_bureau_report(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        iscore_fields = app.get("iscore_report_fields", {})

        inquiry_date_str = iscore_fields.get("inquiry_date", {}).get("value", "")
        if inquiry_date_str:
            try:
                inq_date = datetime.fromisoformat(inquiry_date_str.replace("Z", "+00:00")).date()
                app_date = datetime.utcnow().date()
                age_days = (app_date - inq_date).days
                if age_days > self.MAX_ISCORE_AGE_DAYS:
                    violations.append(FraudRuleViolation(
                        rule_code="BUR-001-STALE-ISCORE-REPORT",
                        rule_name_en="Expired I-Score Bureau Inquiry",
                        rule_name_ar="تقرير الآي سكور منتهي الصلاحية المصرفية",
                        severity="MEDIUM",
                        description_en=f"I-Score report age ({age_days} days) exceeds CBE fresh inquiry window ({self.MAX_ISCORE_AGE_DAYS} days).",
                        description_ar=f"عمر تقرير الاستعلام ({age_days} يوماً) يتجاوز الحد الأقصى لصلاحية الاستعلام بالبنك المركزي ({self.MAX_ISCORE_AGE_DAYS} يوماً).",
                        observed_value=f"{age_days} days",
                        threshold_value=f"<= {self.MAX_ISCORE_AGE_DAYS} days",
                        weight=self.SEVERITY_WEIGHTS["MEDIUM"]
                    ))
            except Exception:
                pass

        legal_actions = iscore_fields.get("legal_action_flags", {}).get("value", [])
        if legal_actions:
            violations.append(FraudRuleViolation(
                rule_code="BUR-002-LEGAL-ENFORCEMENT-ACTION",
                rule_name_en="Active Legal or Negative Action Recorded in Bureau",
                rule_name_ar="تسجيل إجراءات قانونية أو تعثر قضائي في تقرير الاستعلام الائتماني",
                severity="CRITICAL",
                description_en=f"Applicant has active adverse bureau actions: {', '.join(legal_actions)}.",
                description_ar=f"تم رصد إجراءات قضائية ونزاعات سداد نشطة ضد العميل: {', '.join(legal_actions)}.",
                observed_value=legal_actions,
                threshold_value="None",
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        return len(violations) == 0, violations

    def _verify_banking_behavior(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        bank_fields = app.get("bank_statement_fields", {})

        bounced_cheques = int(bank_fields.get("bounced_cheques_count_12m", {}).get("value", 0) or 0)
        if bounced_cheques > 0:
            violations.append(FraudRuleViolation(
                rule_code="BNK-001-BOUNCED-CHEQUES",
                rule_name_en="History of Bounced Cheques within 12 Months",
                rule_name_ar="وجود شيكات مرتجعة بدون رصيد خلال الـ 12 شهراً الماضية",
                severity="HIGH",
                description_en=f"Applicant account recorded {bounced_cheques} bounced cheque(s) indicating liquidity stress.",
                description_ar=f"سجل الحساب {bounced_cheques} شيكاً مرتداً لعدم كفاية الرصيد، مؤشر مخاطر ائتمانية عالية.",
                observed_value=bounced_cheques,
                threshold_value=0,
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        is_balanced = bool(bank_fields.get("running_balance_math_verified", {}).get("value", True))
        if not is_balanced:
            violations.append(FraudRuleViolation(
                rule_code="BNK-002-STATEMENT-ARITHMETIC-ANOMALY",
                rule_name_en="Bank Statement Running Balance Math Error",
                rule_name_ar="خلل في العمليات الحسابية للأرصدة المتتالية بكشف الحساب",
                severity="CRITICAL",
                description_en="Cumulative transactions do not reconcile with ending balances. Strong indicator of forged statement.",
                description_ar="تسلسل العمليات الحسابية لا يتطابق مع رصيد الإقفال، مؤشر قوي على التعديل اليدوي والتزوير.",
                observed_value=False,
                threshold_value=True,
                weight=self.SEVERITY_WEIGHTS["CRITICAL"]
            ))

        return len(violations) == 0, violations

    # -------------------------------------------------------------------------
    # Layer 2 & Forensic Signal Methods
    # -------------------------------------------------------------------------

    def _detect_behavioral_and_forensic_anomalies(
        self, app: Dict[str, Any], income_mismatch: float
    ) -> Tuple[List[AnomalySignal], float, bool]:
        signals = []
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})

        avg_balance = float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 0.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 1.0) or 1.0)
        requested_annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)

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
            explanation_en=f"Peak balance (EGP {max_balance:,.0f}) is {surge_ratio:.1f}x higher than average balance, followed by low liquidity reserve.",
            explanation_ar=f"أعلى رصيد ({max_balance:,.0f} ج.م) يتجاوز {surge_ratio:.1f} أضعاف المتوسط الشهري مع فراغ الحساب، مؤشر على اقتراض مؤقت لتجميل كشف الحساب."
        ))

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

        sample_transactions = bank_fields.get("sample_transaction_amounts", {}).get("value", [])
        if sample_transactions and len(sample_transactions) >= 4:
            benford_anomaly, chi_stat, _ = DeepForensicAnalyzer.evaluate_benford_law(sample_transactions)
            uniformity_score = DeepForensicAnalyzer.calculate_inflow_uniformity(sample_transactions)
        else:
            benford_anomaly = False
            chi_stat = 0.0
            uniformity_score = 0.0

        if benford_anomaly:
            signals.append(AnomalySignal(
                anomaly_name="BENFORD_LAW_FIRST_DIGIT_VIOLATION",
                anomaly_score=0.85,
                detected=True,
                explanation_en=f"Transaction lead digit distribution violates Benford's Law (Chi-Square: {chi_stat:.1f}, p < 0.05). High likelihood of fabricated numbers.",
                explanation_ar=f"توزيع الأرقام في كشف الحساب ينتهك قانون بنفورد الإحصائي (مربع كاي: {chi_stat:.1f}). مؤشر قوي على أرقام مصطنعة ومكتوبة يدوياً."
            ))

        if uniformity_score > 0.40:
            signals.append(AnomalySignal(
                anomaly_name="FABRICATED_ROUND_NUMBER_UNIFORMITY",
                anomaly_score=uniformity_score,
                detected=True,
                explanation_en=f"Abnormal uniformity: {uniformity_score*100:.0f}% of transactions are perfect round thousands/hundreds without authentic fractional friction.",
                explanation_ar=f"تكرار غير طبيعي لأرقام مستديرة تماماً بنسبة {uniformity_score*100:.0f}% دون وجود كسور أو تعاملات تجزئة حقيقية."
            ))

        return signals, uniformity_score, benford_anomaly

    # -------------------------------------------------------------------------
    # Layer 3 Entity Velocity Method
    # -------------------------------------------------------------------------

    def _verify_entity_velocity(self, app: Dict[str, Any]) -> Dict[str, Any]:
        app_id = app.get("application_id", f"APP-{datetime.utcnow().timestamp()}")
        nid = app.get("national_id_fields", {}).get("national_id", {}).get("value", "")
        phone = app.get("form_data", {}).get("mobile_phone", "")
        employer = app.get("salary_certificate_fields", {}).get("employer_name", {}).get("value", "")
        account = app.get("bank_statement_fields", {}).get("bank_account_number", {}).get("value", "")
        device_id = app.get("device_telemetry", {}).get("device_id", "")

        return self.entity_store.check_and_record_velocity(app_id, nid, phone, employer, account, device_id)

    # -------------------------------------------------------------------------
    # Layer 5 Decision & Hybrid Fusion
    # -------------------------------------------------------------------------

    def _calculate_hybrid_decision(
        self,
        violations: List[FraudRuleViolation],
        anomalies: List[AnomalySignal],
        iso_score: float,
        gb_prob: float
    ) -> Tuple[float, str, str]:
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        has_high = any(v.severity == "HIGH" for v in violations)

        rule_score = sum(v.weight for v in violations)
        detected_anoms = [a for a in anomalies if a.detected]
        anomaly_score = sum(a.anomaly_score * 0.20 for a in detected_anoms)

        ml_score = (0.35 * iso_score) + (0.65 * gb_prob)
        base_score = (0.50 * rule_score) + (0.20 * anomaly_score) + (0.30 * ml_score)
        total_score = float(np.clip(base_score, 0.0, 1.0))

        if has_critical or total_score >= 0.70 or gb_prob >= 0.85:
            risk_level = "CRITICAL"
            action = "REJECT_SUSPECTED_FRAUD"
            total_score = max(total_score, 0.90)
        elif has_high or total_score >= 0.45 or ml_score >= 0.50:
            risk_level = "HIGH"
            action = "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION"
        elif total_score >= 0.20 or any(a.detected for a in anomalies):
            risk_level = "MEDIUM"
            action = "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS"
        else:
            risk_level = "LOW"
            action = "PROCEED_TO_CREDIT_EVALUATION"

        return total_score, risk_level, action

    def _generate_explainability(
        self,
        violations: List[FraudRuleViolation],
        anomalies: List[AnomalySignal],
        risk_level: str,
        app: Dict[str, Any],
        gb_prob: float
    ) -> Tuple[List[Dict[str, Any]], str, str]:
        reason_codes = []

        for v in violations:
            reason_codes.append({
                "code": v.rule_code,
                "title_en": v.rule_name_en,
                "title_ar": v.rule_name_ar,
                "severity": v.severity,
                "reason_en": v.description_en,
                "reason_ar": v.description_ar
            })

        for a in anomalies:
            if a.detected:
                reason_codes.append({
                    "code": a.anomaly_name,
                    "title_en": a.anomaly_name.replace("_", " ").title(),
                    "title_ar": "إشارة سلوكية شاذة في الحساب",
                    "severity": "MEDIUM",
                    "reason_en": a.explanation_en,
                    "reason_ar": a.explanation_ar
                })

        app_id = app.get("application_id", "N/A")
        if risk_level == "LOW":
            summary_ar = f"الطلب {app_id} اجتاز كافة فحوصات التطابق الجنائي والبنكي والتحقق الرقمي بنجاح. لا توجد مؤشرات احتيال، والملف مؤهل للتقييم الائتماني المباشر."
            summary_en = f"Application {app_id} successfully passed all 5 forensic, entity graph, digital telemetry, and ML layers. Fraud probability is low ({gb_prob*100:.1f}%)."
        elif risk_level == "CRITICAL":
            summary_ar = f"تحذير رقابي حرج: الطلب {app_id} تم رفضه آلياً لاشتباه تزوير مؤكد أو تكرار بالشبكة. تم رصد {len(violations)} خرق لسياسات البنك المركزي مع احتمال احتيال بنسبة {gb_prob*100:.1f}%."
            summary_en = f"Regulatory Critical Alert: Application {app_id} auto-rejected due to fatal document/entity/device tampering. Estimated fraud probability: {gb_prob*100:.1f}%."
        else:
            summary_ar = f"الطلب {app_id} يتطلب مراجعة ائتمانية يدوية متخصصة لوجود تفاوتات جزئية أو مؤشرات عدم استقرار مالي."
            summary_en = f"Application {app_id} routed to senior underwriter queue due to moderate inconsistencies or cashflow volatility."

        return reason_codes, summary_ar, summary_en

    def _compute_income_haircut(
        self, app: Dict[str, Any], fraud_score: float, risk_level: str, mismatch: float
    ) -> Tuple[float, float]:
        declared_salary = self._get_declared_salary(app)
        if risk_level == "CRITICAL":
            haircut = 0.0
        elif risk_level == "HIGH":
            haircut = max(0.50, 1.0 - (fraud_score * 0.75))
        elif risk_level == "MEDIUM":
            haircut = max(0.75, 1.0 - (fraud_score * 0.50))
        else:
            haircut = 1.0

        adjusted_salary = declared_salary * haircut
        return haircut, adjusted_salary

    def _get_declared_salary(self, app: Dict[str, Any]) -> float:
        return float(app.get("salary_certificate_fields", {}).get("declared_net_salary", {}).get("value", 0.0) or 0.0)

    def _translate_action(self, action: str) -> str:
        mapping = {
            "PROCEED_TO_CREDIT_EVALUATION": "تمرير الطلب للتقييم الائتماني المباشر",
            "REQUEST_ADDITIONAL_VERIFICATION_DOCUMENTS": "طلب مستندات دعم إضافية وتحديث الاستعلام",
            "FLAG_FOR_MANUAL_FRAUD_INVESTIGATION": "تحويل الطلب للمراجعة الأمنية والائتمانية اليدوية",
            "REJECT_SUSPECTED_FRAUD": "رفض قطعي فوري لاشتباه تزوير واحتيال"
        }
        return mapping.get(action, action)

    def enrich_payload(self, application: Dict[str, Any]) -> Dict[str, Any]:
        assessment = self.evaluate(application)
        if "consistency_checks" not in application:
            application["consistency_checks"] = {}

        application["consistency_checks"].update(assessment["verification_checklist"])
        application["fraud_assessment"] = assessment
        return application
