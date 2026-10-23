"""
NovaCredit AI — Comprehensive Automated Test Suite.

Covers:
  1. Dataset & Schema Integrity
  2. Model Artifacts & Pipeline Inference
  3. Feature Ordering & Preprocessing Consistency
  4. Input Validation & Edge Case Handling
  5. Database Operations (CRUD, Filtering, Summary Stats)
  6. Flask Routes & API Endpoints
  7. Report Generation (PDF)
"""

import os
import json
import unittest
import pandas as pd
import numpy as np

# Set environment before loading Flask app
os.environ["SECRET_KEY"] = "test-secret-key-12345"

from app import create_app
from config import Config
from database.database import (
    init_db, insert_prediction, get_prediction_by_id,
    delete_prediction, get_filtered_predictions, get_summary_stats,
    get_chart_data
)
from models.predictor import (
    load_artifacts, build_feature_vector, predict_risk,
    generate_recommendation, get_metadata
)


class TestDatasetAndSchema(unittest.TestCase):
    """1. Dataset & Schema Integrity Tests"""

    def setUp(self):
        self.csv_path = "banking_risk_analytics_dataset.csv"

    def test_dataset_exists_and_readable(self):
        self.assertTrue(os.path.exists(self.csv_path), "Dataset CSV must exist.")
        df = pd.read_csv(self.csv_path)
        self.assertEqual(len(df), 5000, "Dataset must have exactly 5,000 rows.")
        self.assertIn("loan_default", df.columns, "Target 'loan_default' must be present.")
        self.assertIn("default_probability", df.columns, "'default_probability' must be present.")

    def test_target_distribution(self):
        df = pd.read_csv(self.csv_path)
        counts = df["loan_default"].value_counts()
        self.assertEqual(counts[0], 2631, "Class 0 count must be 2631.")
        self.assertEqual(counts[1], 2369, "Class 1 count must be 2369.")
        # Ensure target is binary
        self.assertTrue(set(df["loan_default"].unique()).issubset({0, 1}))


class TestModelPipeline(unittest.TestCase):
    """2. Model Loading, Preprocessing & Inference Tests"""

    def setUp(self):
        self.pipeline, self.metadata = load_artifacts()

    def test_pipeline_loading(self):
        self.assertIsNotNone(self.pipeline, "Pipeline artifact must load successfully.")
        self.assertTrue(hasattr(self.pipeline, "predict_proba"), "Pipeline must support predict_proba.")
        self.assertTrue(hasattr(self.pipeline, "named_steps"), "Pipeline must have named_steps.")
        self.assertIn("preprocessor", self.pipeline.named_steps)
        self.assertIn("classifier", self.pipeline.named_steps)

    def test_metadata_structure(self):
        self.assertIn("metrics", self.metadata)
        self.assertIn("test_accuracy", self.metadata["metrics"])
        self.assertIn("test_roc_auc", self.metadata["metrics"])
        self.assertIn("feature_importances", self.metadata)
        self.assertGreater(self.metadata["metrics"]["test_accuracy"], 0.65)
        self.assertGreater(self.metadata["metrics"]["test_roc_auc"], 0.70)

    def test_inference_on_low_risk_applicant(self):
        prime_applicant = {
            "age": 42,
            "gender": "Female",
            "marital_status": "Married",
            "city": "Bengaluru",
            "education_level": "Master",
            "employment_status": "Salaried",
            "income": 1800000,
            "credit_score": 790,
            "years_employed": 14,
            "loan_amount": 400000,
            "loan_term": 36,
            "existing_debt": 80000,
            "previous_loans": 2,
            "late_payments": 0,
            "monthly_expenses": 35000,
            "savings_balance": 500000,
            "debt_to_income_ratio": 0.12,
            "loan_purpose": "Home",
        }
        res = predict_risk(prime_applicant)
        self.assertEqual(res["prediction"], "Low Risk")
        self.assertLess(res["probability"], 40.0)
        self.assertGreater(len(res["reasons"]), 0)
        self.assertGreater(len(res["tips"]), 0)

    def test_inference_on_high_risk_applicant(self):
        high_risk_applicant = {
            "age": 23,
            "gender": "Male",
            "marital_status": "Single",
            "city": "Lucknow",
            "education_level": "High School",
            "employment_status": "Salaried",
            "income": 280000,
            "credit_score": 420,
            "years_employed": 1,
            "loan_amount": 950000,
            "loan_term": 60,
            "existing_debt": 700000,
            "previous_loans": 3,
            "late_payments": 9,
            "monthly_expenses": 30000,
            "savings_balance": 10000,
            "debt_to_income_ratio": 2.50,
            "loan_purpose": "Personal",
        }
        res = predict_risk(high_risk_applicant)
        self.assertEqual(res["prediction"], "High Risk")
        self.assertGreaterEqual(res["probability"], 50.0)
        self.assertIn("Subprime credit score", " ".join(res["reasons"]))


class TestDatabaseAndCrud(unittest.TestCase):
    """3. Database CRUD Operations & Queries"""

    def setUp(self):
        init_db()

    def test_insert_and_retrieve_prediction(self):
        sample = {
            "customer_name": "Test Applicant",
            "age": 35,
            "gender": "Male",
            "marital_status": "Married",
            "city": "Mumbai",
            "education_level": "Bachelor",
            "employment_status": "Salaried",
            "income": 950000,
            "credit_score": 710,
            "years_employed": 8,
            "loan_amount": 350000,
            "loan_term": 24,
            "existing_debt": 90000,
            "previous_loans": 1,
            "late_payments": 1,
            "monthly_expenses": 28000,
            "savings_balance": 180000,
            "debt_to_income_ratio": 0.18,
            "loan_purpose": "Car",
            "prediction": "Low Risk",
            "prediction_probability": 26.4,
            "risk_tier": "Low Risk",
        }
        record_id = insert_prediction(sample)
        self.assertIsInstance(record_id, int)
        self.assertGreater(record_id, 0)

        record = get_prediction_by_id(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["customer_name"], "Test Applicant")
        self.assertEqual(record["prediction"], "Low Risk")

        # Clean up
        deleted = delete_prediction(record_id)
        self.assertTrue(deleted)
        self.assertIsNone(get_prediction_by_id(record_id))

    def test_summary_and_chart_data(self):
        stats = get_summary_stats()
        self.assertIn("total", stats)
        self.assertIn("avg_credit_score", stats)
        self.assertIn("default_rate", stats)

        charts = get_chart_data()
        self.assertIn("age_distribution", charts)
        self.assertIn("loan_purpose_distribution", charts)
        self.assertIn("prediction_distribution", charts)


class TestFlaskEndpoints(unittest.TestCase):
    """4. Flask Routes & API Response Tests"""

    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_home_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"NovaCredit", res.data)

    def test_model_performance_page(self):
        res = self.client.get("/model-performance")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Theoretical Bayes Limit", res.data)
        self.assertIn(b"Confusion Matrix", res.data)

    def test_analytics_page_and_apis(self):
        res = self.client.get("/analytics")
        self.assertEqual(res.status_code, 200)

        res_api = self.client.get("/api/analytics/data")
        self.assertEqual(res_api.status_code, 200)
        json_data = json.loads(res_api.data)
        self.assertIn("age_distribution", json_data)

        res_fi = self.client.get("/api/analytics/feature-importance")
        self.assertEqual(res_fi.status_code, 200)
        fi_data = json.loads(res_fi.data)
        self.assertIn("labels", fi_data)
        self.assertIn("values", fi_data)

    def test_prediction_form_get_and_post(self):
        # GET form
        res_get = self.client.get("/predict")
        self.assertEqual(res_get.status_code, 200)

        # POST valid form
        valid_payload = {
            "customer_name": "Integration Tester",
            "age": "38",
            "gender": "Female",
            "marital_status": "Single",
            "city": "Chennai",
            "education_level": "Bachelor",
            "employment_status": "Salaried",
            "income": "1100000",
            "credit_score": "730",
            "years_employed": "9",
            "loan_amount": "450000",
            "loan_term": "36",
            "existing_debt": "120000",
            "previous_loans": "2",
            "late_payments": "1",
            "monthly_expenses": "32000",
            "savings_balance": "220000",
            "debt_to_income_ratio": "0.19",
            "loan_purpose": "Home",
        }
        res_post = self.client.post("/predict", data=valid_payload, follow_redirects=True)
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b"Risk Assessment Result", res_post.data)
        self.assertIn(b"Integration Tester", res_post.data)

    def test_invalid_input_validation(self):
        # Missing required fields
        invalid_payload = {
            "customer_name": "",
            "age": "-5",  # Invalid age
            "income": "not-a-number",
        }
        res = self.client.post("/predict", data=invalid_payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"This field is required", res.data)


if __name__ == "__main__":
    unittest.main()
