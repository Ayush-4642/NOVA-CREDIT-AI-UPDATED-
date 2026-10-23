"""
Application configuration for the NovaCredit AI.
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "credit-risk-analytics-secret-key")

    # Base directory
    BASE_DIR = BASE_DIR

    # Paths
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "banking.db")
    MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
    PIPELINE_PATH = os.path.join(BASE_DIR, "models", "credit_risk_pipeline.pkl")
    METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
    SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

    # Pagination
    RECORDS_PER_PAGE = 10

    # Production Model Features (Single Source of Truth)
    NUMERIC_FEATURES = [
        "age",
        "income",
        "credit_score",
        "years_employed",
        "loan_amount",
        "loan_term_months",
        "existing_debt",
        "previous_loans",
        "late_payments",
        "monthly_expenses",
        "savings_balance",
        "debt_to_income_ratio",
    ]

    CATEGORICAL_FEATURES = [
        "gender",
        "marital_status",
        "city",
        "education_level",
        "employment_status",
        "loan_purpose",
    ]

    MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

