"""
Senior Production-Grade Credit Application Fraud & Cross-Document Consistency Engine
=====================================================================================
Platform: Smart Financing & Credit Request Analysis Platform (ZAWOLF)
Standard: Egyptian Banking Federation & CBE Credit Risk Compliance Guidelines
Architecture:
  - Layer 1: Deterministic Cross-Document Rules Engine (OCR + Bureau + Core Banking)
  - Layer 2: Behavioral & Statistical Anomaly Detection (Window Dressing, Volatility)
  - Layer 3: Explainable AI (XAI) & Regulatory Reason Codes (Bilingual AR/EN)
  - Layer 4: Credit Risk Discount & Income Haircut Calculator (Downstream Feeder)
"""

import json
import re
import math
import difflib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict


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


class CreditFraudEngine:
    """
    Senior / Production-Grade Hybrid Fraud & Cross-Document Consistency Engine.
    Processes Application JSON payloads, computes rule violations, statistical anomalies,
    generates XAI Reason Codes, and calculates income haircut multipliers for Risk models.
    """

    SEVERITY_WEIGHTS = {
        "LOW": 0.10,
        "MEDIUM": 0.25,
        "HIGH": 0.50,
        "CRITICAL": 0.90
    }

    # Operational & Regulatory Thresholds
    INCOME_MISMATCH_WARN = 0.20        # 20% discrepancy triggers warning
    INCOME_MISMATCH_CRITICAL = 0.40    # 40% discrepancy triggers critical alert
    EMPLOYER_SIMILARITY_MIN = 0.65     # Minimum string similarity for employer
    MAX_ISCORE_AGE_DAYS = 30           # 30-day regulatory fresh inquiry limit
    MIN_APPLICANT_AGE = 21             # Legal financing age in Egypt
    MAX_APPLICANT_AGE = 65             # Retirement threshold
    MIN_OCR_CONFIDENCE_THRESHOLD = 0.70
    WINDOW_DRESSING_SURGE_RATIO = 2.5  # Max balance > 2.5x average indicates artificial pump
    ANNUITY_TO_LIQUIDITY_MAX = 0.60    # Loan annuity > 60% of average balance is high risk

    def __init__(self):
        pass

    def evaluate(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end 4-layer evaluation on the application JSON payload.
        """
        violations: List[FraudRuleViolation] = []
        checks_passed: Dict[str, bool] = {}

        # ---------------------------------------------------------------------
        # Layer 1: Deterministic Cross-Document Rules
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

        # 5. I-Score Bureau Freshness & Facilities
        bureau_ok, bur_v = self._verify_bureau_report(application)
        checks_passed["bureau_verified"] = bureau_ok
        violations.extend(bur_v)

        # 6. Bank Statement Math Reconciliation
        bank_ok, bank_v = self._verify_banking_behavior(application)
        checks_passed["bank_statement_math_verified"] = bank_ok
        violations.extend(bank_v)

        # ---------------------------------------------------------------------
        # Layer 2: Behavioral & Statistical Anomaly Detection
        # ---------------------------------------------------------------------
        anomaly_signals = self._detect_behavioral_anomalies(application, income_mismatch)

        # ---------------------------------------------------------------------
        # Layer 3: Risk Scoring & Decision
        # ---------------------------------------------------------------------
        fraud_score, risk_level, action = self._calculate_decision(violations, anomaly_signals)

        # ---------------------------------------------------------------------
        # Layer 4: XAI Reason Codes & Downstream Credibility Haircut
        # ---------------------------------------------------------------------
        reason_codes, executive_summary_ar, executive_summary_en = self._generate_explainability(
            violations, anomaly_signals, risk_level, application
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
                "detected_anomalies_count": sum(1 for a in anomaly_signals if a.detected)
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
        Enriches the input JSON payload by updating consistency_checks
        and appending the full fraud_assessment block.
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

    # -------------------------------------------------------------------------
    # Layer 1 Verification Details
    # -------------------------------------------------------------------------

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
        salary_fields = app.get("salary_certificate_fields", {})
        employer_name = str(salary_fields.get("employer_name", {}).get("value", "")).strip()

        precomputed_similarity = app.get("consistency_checks", {}).get("employer_match_similarity_score")
        if precomputed_similarity is not None:
            similarity = float(precomputed_similarity)
        elif employer_name:
            similarity = 1.0
        else:
            similarity = 0.0

        if not employer_name:
            violations.append(FraudRuleViolation(
                rule_code="RULE_MISSING_EMPLOYER_NAME",
                rule_name_en="Missing Employer Name on Pay Slip",
                rule_name_ar="اسم جهة العمل مفقود أو غير مقروء في شهادة الراتب",
                severity="HIGH",
                description_en="Employer name could not be extracted from salary certificate.",
                description_ar="تعذر استخراج اسم جهة العمل من شهادة مفردات المرتب المرفوعة.",
                observed_value="",
                threshold_value="Valid Company Name",
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))
        elif similarity < self.EMPLOYER_SIMILARITY_MIN:
            violations.append(FraudRuleViolation(
                rule_code="RULE_EMPLOYER_NAME_MISMATCH",
                rule_name_en="Employer Name Mismatch in Payroll Inflows",
                rule_name_ar="اختلاف اسم جهة العمل بين شهادة الراتب والجهة المحولة بنكياً",
                severity="HIGH",
                description_en=(f"Employer on salary certificate ('{employer_name}') does not match "
                                f"payroll originator on bank statement (similarity {similarity*100:.1f}%)."),
                description_ar=(f"اسم الشركة في شهادة الراتب ('{employer_name}') لا يتطابق مع الجهة المحولة في البنك "
                                f"(نسبة التطابق {similarity*100:.1f}% فقط)."),
                observed_value=round(similarity, 3),
                threshold_value=self.EMPLOYER_SIMILARITY_MIN,
                weight=self.SEVERITY_WEIGHTS["HIGH"]
            ))

        return len(violations) == 0, violations, similarity

    def _verify_document_integrity(self, app: Dict[str, Any]) -> Tuple[bool, List[FraudRuleViolation]]:
        violations = []
        for doc in app.get("documents", []):
            doc_type = doc.get("document_type", "unknown")
            quality = float(doc.get("overall_quality_score", 1.0))
            is_tampered = bool(doc.get("is_tampered_suspected", False))

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

    # -------------------------------------------------------------------------
    # Layer 2: Behavioral & Statistical Anomaly Detection
    # -------------------------------------------------------------------------

    def _detect_behavioral_anomalies(self, app: Dict[str, Any], income_mismatch: float) -> List[AnomalySignal]:
        signals = []
        bank_fields = app.get("bank_statement_fields", {})
        form_data = app.get("form_data", {})

        avg_balance = float(bank_fields.get("avg_monthly_balance", {}).get("value", 0.0) or 0.0)
        max_balance = float(bank_fields.get("max_monthly_balance", {}).get("value", 0.0) or 0.0)
        min_balance = float(bank_fields.get("min_monthly_balance", {}).get("value", 0.0) or 0.0)
        volatility = float(bank_fields.get("balance_volatility_std", {}).get("value", 0.0) or 0.0)
        regularity = float(bank_fields.get("income_regularity_score", {}).get("value", 1.0) or 1.0)
        requested_annuity = float(form_data.get("requested_annuity", 0.0) or 0.0)

        # 1. Window Dressing Detector (Artificial balance inflation right before applying)
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
            explanation_en=f"Peak balance ({max_balance:,.0f}) is {surge_ratio:.1f}x higher than average balance ({avg_balance:,.0f}), followed by low liquidity reserve.",
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

        return signals

    # -------------------------------------------------------------------------
    # Layer 3: Risk Scoring & Decision Engine
    # -------------------------------------------------------------------------

    def _calculate_decision(self, violations: List[FraudRuleViolation], anomalies: List[AnomalySignal]) -> Tuple[float, str, str]:
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        has_high = any(v.severity == "HIGH" for v in violations)

        # Compute base score from rules
        rule_score = sum(v.weight for v in violations)
        
        # Add contribution from detected statistical anomalies
        anomaly_score = sum(a.anomaly_score * 0.20 for a in anomalies if a.detected)
        
        composite_score = min(0.05 + rule_score + anomaly_score, 1.0)

        if has_critical or composite_score >= 0.75:
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

    # -------------------------------------------------------------------------
    # Layer 4: XAI Reason Codes & Downstream Credibility Haircut
    # -------------------------------------------------------------------------

    def _generate_explainability(self, violations: List[FraudRuleViolation], anomalies: List[AnomalySignal],
                                risk_level: str, app: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str, str]:
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

        # Generate Executive Narrative
        app_id = app.get("application_id", "N/A")
        if risk_level == "LOW":
            summary_ar = f"الطلب {app_id} اجتاز كافة فحوصات التطابق الجنائي والبنكي بنجاح. لا توجد أي مؤشرات تلاعب، والملف جاهز للتقييم الائتماني المباشر."
            summary_en = f"Application {app_id} successfully passed all forensic and cross-document integrity checks. Zero fraud indicators detected."
        elif risk_level == "CRITICAL":
            summary_ar = f"تحذير رقابي: الطلب {app_id} مرفوض لاشتباه تزوير عالي الخطورة. تم رصد {len(violations)} مخالفة حتمية تستدعي الرفض الفوري وتوثيق الحالة في سجل المخاطر التشغيلية."
            summary_en = f"Regulatory Alert: Application {app_id} flagged for critical fraud suspicion. {len(violations)} fatal violation(s) identified requiring immediate auto-rejection."
        else:
            summary_ar = f"الطلب {app_id} يتطلب عناية ومراجعة يدوية من مسؤول الائتمان قبل المنح بسبب وجود تضارب جزئي أو سلوك مصرفي شاذ في كشف الحساب."
            summary_en = f"Application {app_id} requires senior manual underwriter review due to moderate inconsistencies or cashflow anomalies."

        return reason_codes, summary_ar, summary_en

    def _compute_income_haircut(self, app: Dict[str, Any], fraud_score: float, risk_level: str, mismatch: float) -> Tuple[float, float]:
        """
        Calculates a Credibility Discount Multiplier for the downstream Risk Model.
        If fraud score is moderate, we apply an income haircut (e.g. 15-30%) rather than a binary rejection.
        """
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
