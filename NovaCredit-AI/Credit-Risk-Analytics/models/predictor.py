"""
NovaCredit AI — Production Prediction Service.

Wraps the unified Scikit-Learn Pipeline artifact:
    - Preprocessing: ColumnTransformer (StandardScaler + OneHotEncoder)
    - Estimator: Tuned Regularized RandomForestClassifier
    - Metadata: model_metadata.json (verified evaluation metrics & feature schemas)

Provides single-source-of-truth inference matching Model_Training.ipynb exactly.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from config import Config

_pipeline = None
_metadata = None


def load_artifacts():
    """Load pipeline artifact and evaluation metadata (singleton pattern)."""
    global _pipeline, _metadata

    if _pipeline is None:
        # Load unified pipeline
        pipeline_path = Config.MODEL_PATH
        if not os.path.exists(pipeline_path):
            alt_path = os.path.join(Config.BASE_DIR, "models", "credit_risk_pipeline.pkl")
            if os.path.exists(alt_path):
                pipeline_path = alt_path
        _pipeline = joblib.load(pipeline_path)

    if _metadata is None:
        meta_path = os.path.join(Config.BASE_DIR, "models", "model_metadata.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(Config.BASE_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        else:
            _metadata = {}

    return _pipeline, _metadata


def get_metadata():
    """Returns the verified evaluation metadata dictionary."""
    _, metadata = load_artifacts()
    return metadata


def build_feature_vector(form_data: dict) -> pd.DataFrame:
    """
    Builds the feature DataFrame with all expected numeric and categorical features.
    Handles field name aliases (such as 'loan_term' -> 'loan_term_months').
    """
    _, metadata = load_artifacts()
    
    numeric_features = metadata.get("features", {}).get("numeric_features", [
        "age", "income", "credit_score", "years_employed", "loan_amount",
        "loan_term_months", "existing_debt", "previous_loans", "late_payments",
        "monthly_expenses", "savings_balance", "debt_to_income_ratio"
    ])
    categorical_features = metadata.get("features", {}).get("categorical_features", [
        "gender", "marital_status", "city", "education_level", "employment_status", "loan_purpose"
    ])
    
    row = {}

    # Extract numeric fields
    for col in numeric_features:
        # Check alias
        val = form_data.get(col)
        if val is None and col == "loan_term_months":
            val = form_data.get("loan_term", 0)
        try:
            row[col] = [float(val) if val not in (None, "") else 0.0]
        except (ValueError, TypeError):
            row[col] = [0.0]

    # Extract categorical fields
    for col in categorical_features:
        val = form_data.get(col)
        row[col] = [str(val).strip() if val not in (None, "") else "Unknown"]

    all_cols = numeric_features + categorical_features
    return pd.DataFrame(row, columns=all_cols)


def generate_recommendation(prediction: str, form_data: dict, probability: float = None):
    """
    Generates explainable credit risk drivers and actionable recommendations
    based on the applicant's financial attributes.
    """
    reasons = []
    tips = []

    try:
        credit_score = float(form_data.get("credit_score", 0))
        existing_debt = float(form_data.get("existing_debt", 0))
        dti = float(form_data.get("debt_to_income_ratio", 0))
        late_payments = float(form_data.get("late_payments", 0))
        savings = float(form_data.get("savings_balance", 0))
        income = float(form_data.get("income", 0))
        loan_amount = float(form_data.get("loan_amount", 0))
    except (ValueError, TypeError):
        credit_score = existing_debt = dti = late_payments = savings = income = loan_amount = 0

    if prediction == "High Risk":
        if credit_score < 580:
            reasons.append(f"Subprime credit score ({int(credit_score)} vs recommended 650+)")
            tips.append("Focus on rebuilding credit history with secured credit and zero missed payments.")
        elif credit_score < 660:
            reasons.append(f"Fair credit score ({int(credit_score)}) indicates elevated default probability.")
            tips.append("Aim for credit score above 700 to qualify for prime lending rates.")

        if late_payments >= 5:
            reasons.append(f"Significant history of late payments ({int(late_payments)} recorded)")
            tips.append("Establish automated bill payments to prevent delinquent payment flags.")
        elif late_payments >= 2:
            reasons.append(f"Multiple late payment instances ({int(late_payments)} noted)")
            tips.append("Maintain 6-12 consecutive months of on-time payments to improve risk scoring.")

        if dti > 0.60:
            reasons.append(f"High debt-to-income ratio ({dti:.2f}) indicates severe leverage.")
            tips.append("Consolidate or pay down existing liabilities to bring DTI below 0.40.")
        elif dti > 0.43:
            reasons.append(f"Elevated debt-to-income ratio ({dti:.2f}) exceeds standard banking threshold.")
            tips.append("Reduce short-term debt before taking on additional loan obligations.")

        if income > 0 and existing_debt > (income * 0.8):
            reasons.append(f"Existing debt (Rs. {existing_debt:,.0f}) is high relative to annual income (Rs. {income:,.0f}).")
            tips.append("Amortize high-interest existing debt before applying for fresh credit.")

        if income > 0 and savings < (income * 0.05):
            reasons.append("Minimal emergency liquid savings buffer relative to declared income.")
            tips.append("Build a cash reserve covering at least 3-6 months of essential expenses.")

        if not reasons:
            reasons.append("Composite multivariate risk profile flagged by ensemble decision trees.")
            tips.append("Review overall liability structure and re-apply with a co-signer or collateral.")
    else:
        # Low Risk
        if credit_score >= 750:
            reasons.append(f"Excellent credit score ({int(credit_score)}) demonstrating prime reliability.")
        elif credit_score >= 670:
            reasons.append(f"Good credit score ({int(credit_score)}) reflecting stable credit management.")
        else:
            reasons.append("Credit score within acceptable institutional risk boundaries.")

        if dti <= 0.35:
            reasons.append(f"Healthy and sustainable debt-to-income ratio ({dti:.2f}).")
        
        if late_payments == 0:
            reasons.append("Spotless on-time payment track record.")
        elif late_payments <= 2:
            reasons.append(f"Manageable repayment history with low delinquency incidents ({int(late_payments)}).")

        if savings >= loan_amount * 0.2:
            reasons.append("Robust liquid savings buffer mitigating default exposure.")

        tips.append("Maintain current financial discipline and timely repayments.")
        tips.append("Eligible for competitive prime interest rates and flexible loan tenures.")

    return reasons, tips


def predict_risk(form_data: dict) -> dict:
    """
    Executes full prediction workflow:
      1. Prepares verified feature vector matching ColumnTransformer.
      2. Invokes pipeline to get calibrated probability of default.
      3. Assigns risk tier and label.
      4. Synthesizes explainable reasons and recommendations.
    """
    pipeline, _ = load_artifacts()

    features_df = build_feature_vector(form_data)
    proba = pipeline.predict_proba(features_df)[0]

    # Index 1 corresponds to loan_default == 1 (High Risk)
    classes = list(pipeline.classes_)
    default_index = classes.index(1) if 1 in classes else 1
    prob_default = float(proba[default_index])

    # Standard threshold 0.50
    prediction_label = "High Risk" if prob_default >= 0.50 else "Low Risk"
    confidence = prob_default if prediction_label == "High Risk" else (1.0 - prob_default)

    # Granular Risk Tier
    if prob_default >= 0.75:
        risk_tier = "Critical Risk"
    elif prob_default >= 0.50:
        risk_tier = "High Risk"
    elif prob_default >= 0.35:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "Low Risk"

    reasons, tips = generate_recommendation(prediction_label, form_data, prob_default)

    return {
        "prediction": prediction_label,
        "probability": round(prob_default * 100, 2),
        "confidence": round(confidence * 100, 2),
        "risk_tier": risk_tier,
        "reasons": reasons,
        "tips": tips,
    }
