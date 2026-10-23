# 💠 NovaCredit AI

### 🏦 AI-Powered Banking Credit Risk Intelligence & Prediction Platform

⭐ **If you like this project, give it a star — it means a lot!** ⭐

🚀 **Live Demos:**

- 🚀 [NovaCredit AI — Latest Deployment](https://nova-credit-ai-updated-2.onrender.com)

💻 **GitHub Repository:** [NOVA-CREDIT-AI-UPDATED](https://github.com/Ayush-4642/NOVA-CREDIT-AI-UPDATED-)


---

## 📚 Table of Contents

- [✨ Why NovaCredit AI?](#-why-novacredit-ai)
- [🚀 Features](#-features)
- [🧠 How the ML Pipeline Works](#-how-the-ml-pipeline-works)
- [⚠️ Prediction Integrity](#️-prediction-integrity)
- [🛠️ Tech Stack](#️-tech-stack)
- [🏗️ System Architecture](#️-system-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Deployment on Render](#-deployment-on-render)
- [▶️ Usage](#️-usage)
- [🧪 Testing](#-testing)
- [📊 Model Performance](#-model-performance)
- [🎯 Decision Threshold Analysis](#-decision-threshold-analysis)
- [🔮 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## ✨ Why NovaCredit AI?

Traditional lending institutions need reliable ways to evaluate borrower risk, identify potentially risky applications, and monitor their loan portfolios.

**NovaCredit AI** is a full-stack machine learning application that transforms a credit risk prediction model into an interactive banking analytics platform.

Users can enter applicant information, receive a predicted risk classification, explore risk factors, review prediction history, and analyze portfolio-level statistics through an interactive dashboard.

The project combines machine learning, backend development, database management, data visualization, and automated reporting in one application.

> 💡 **Project vision:** Turn a data science workflow into a practical FinTech application — from applicant information and model inference to risk reporting, historical records, and portfolio analytics.

---

## 🚀 Features

### 🎯 Credit Risk Prediction

- Guided, multi-step loan application form.
- Applicant information validation.
- Machine learning-based risk classification.
- Risk probability and risk category presentation.
- Plain-language risk factors and recommendations.

### 📊 Interactive Analytics

- Portfolio-level risk distribution.
- Customer demographic analysis.
- Credit risk analytics and visualizations.
- Model performance and validation dashboard.
- Interactive charts powered by Chart.js.

### 🗂️ Prediction History & Reporting

- Store prediction records in SQLite.
- Search, filter, and review historical assessments.
- Export prediction history to CSV and PDF.
- Generate individual PDF risk reports.
- View printable loan evaluation reports.

### 🛡️ Administrative Dashboard

- Portfolio monitoring.
- Key risk indicators.
- Recent prediction activity.
- Aggregated customer and lending statistics.

### 🎨 Modern User Interface

- FinTech-inspired glassmorphism design.
- Dark and light themes.
- Responsive layouts for desktop, tablet, and mobile.
- Interactive forms and visual risk indicators.

---

## 🧠 How the ML Pipeline Works

The application follows a machine learning inference workflow:

```text
📝 Applicant Information
          |
          v
✅ Input Validation
          |
          v
🔄 Feature Preparation
          |
          v
🌲 Trained Random Forest Pipeline
          |
          v
📊 Risk Classification & Probability
          |
          v
💡 Risk Factors & Recommendations
          |
          v
💾 SQLite Prediction History
          |
          v
📈 Analytics & PDF Reporting
```

### Pipeline Overview

1. **Input collection:** The applicant provides the required financial, demographic, and loan-related information.
2. **Validation:** The application checks submitted values before prediction.
3. **Preprocessing:** Features are prepared using the same preprocessing pipeline used during model training.
4. **Prediction:** The trained model generates a risk classification and associated probability.
5. **Interpretation:** The application presents risk information and recommendations.
6. **Storage:** Prediction details are stored in the database.
7. **Reporting:** Historical predictions contribute to analytics and reporting features.

---

## ⚠️ Prediction Integrity

The deployed application should use the same preprocessing and feature schema as the trained model.

The v2.1 model documentation describes a unified Scikit-learn pipeline containing a `ColumnTransformer`, numeric feature scaling, categorical encoding, and a Random Forest classifier.

This design helps keep training and prediction preprocessing consistent.

**Important implementation principles:**

- Load the serialized model trained for the deployed application.
- Preserve the expected feature names and data types.
- Use the same preprocessing steps used during training.
- Handle missing or invalid input before inference.
- Keep model artifacts and metadata synchronized.
- Never assume that a displayed probability is perfectly calibrated unless calibration has been evaluated.

The exact deployed artifact should be checked against the model version and metrics documented in this repository.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, JavaScript |
| UI Templates | Flask, Jinja2 |
| Backend | Python, Flask |
| Machine Learning | Scikit-learn |
| Model Persistence | Pickle / serialized model pipeline |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Visualizations | Chart.js |
| PDF Reports | ReportLab |
| Deployment | Render |
| Version Control | Git, GitHub |
| Testing | Python unittest |

---

## 🏗️ System Architecture

```text
          User / Browser
                |
                v
       HTML + CSS + JavaScript
                |
                v
          Flask Application
                |
       +--------+---------+
       |        |         |
       v        v         v
    Routes   Predictor   Database
       |        |         |
       |        v         |
       |   Trained ML     |
       |     Pipeline     |
       |        |         |
       +--------+---------+
                |
       +--------+---------+
       |                  |
       v                  v
  Analytics &         PDF / CSV
  Dashboards           Reports
```

The Flask backend connects the web interface to the prediction pipeline, database operations, analytics, and reporting utilities.

---

## 📁 Project Structure

The following structure reflects the supplied project documentation. Check your actual repository before relying on individual filenames or paths.

```text
NOVA-CREDIT-AI-UPDATED/
└── Credit-Risk-Analytics/
    ├── app.py
    ├── config.py
    ├── requirements.txt
    ├── train_model.py
    ├── banking_risk_analytics_dataset.csv
    ├── random_forest_model.pkl
    ├── model_metadata.json
    ├── benchmark_results.json
    ├── EDA.ipynb
    ├── Model_Training.ipynb
    │
    ├── models/
    │   ├── credit_risk_pipeline.pkl
    │   ├── model_metadata.json
    │   └── predictor.py
    │
    ├── database/
    │   └── database.py
    │
    ├── routes/
    │   ├── home.py
    │   ├── predict.py
    │   ├── analytics.py
    │   ├── dashboard.py
    │   ├── history.py
    │   └── admin.py
    │
    ├── utils/
    │   └── pdf_report.py
    │
    ├── templates/
    │   ├── base.html
    │   ├── home.html
    │   ├── predict.html
    │   ├── result.html
    │   ├── model_performance.html
    │   ├── analytics.html
    │   ├── dashboard.html
    │   ├── history.html
    │   ├── admin.html
    │   ├── report.html
    │   └── about.html
    │
    ├── static/
    │   ├── css/
    │   │   └── style.css
    │   └── js/
    │       ├── script.js
    │       └── charts.js
    │
    ├── tests/
    │   └── test_app.py
    │
    ├── reports/
    └── exports/
```
---

## 🚀 Deployment on Render

NovaCredit AI can be deployed as a Python web service on Render.

### Deployment Steps

1. Push your project to GitHub.
2. Open the [Render Dashboard](https://dashboard.render.com/).
3. Select **New → Web Service**.
4. Connect your GitHub repository.
5. Select the branch containing the application.
6. Configure the correct root directory.
7. Enter the build and start commands.
8. Select an appropriate instance type.
9. Deploy and inspect the build and runtime logs.

### Render Configuration

If `app.py` and `requirements.txt` are located in the selected service's root directory, and `app.py` exposes a Flask object named `app`, use:

**Language**

```text
Python 3
```

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

**Root Directory**

Set this to `Credit-Risk-Analytics` if that directory contains the application's entry point and dependency file. Otherwise, leave it blank when the files are at the repository root.

### Deployment Checklist

- [ ] `app.py` is present in the configured root directory.
- [ ] `requirements.txt` is present in that same directory.
- [ ] Gunicorn is included in the dependencies.
- [ ] Required model artifacts are available.
- [ ] Python and library versions are compatible.
- [ ] SQLite database initialization works in the deployed environment.
- [ ] The application binds to Render's assigned port.
- [ ] The deployed website and prediction workflow have been tested.

**Note:** Render's free services may spin down after inactivity. Local filesystem changes and SQLite data on an ephemeral filesystem may not persist across restarts or redeployments. Use persistent storage or a managed database if durable production data is required.

---

## ▶️ Usage

After launching the application:

1. Open the home page.
2. Navigate to the prediction form.
3. Enter the required applicant and loan information.
4. Submit the form to obtain a risk assessment.
5. Review the risk classification, probability, and recommendations.
6. Open the analytics dashboard to explore portfolio statistics.
7. Review previous assessments in prediction history.
8. Export available reports as PDF or CSV.

Available routes depend on the routes registered by the deployed application.

| Route | Purpose |
|---|---|
| `/` | Home page |
| `/predict` | Submit a credit risk assessment |
| `/predict/result/<id>` | View a prediction result |
| `/predict/report/<id>` | View or download a prediction report |
| `/history` | Review historical assessments |
| `/analytics` | Explore portfolio analytics |
| `/dashboard` | View customer statistics |
| `/admin` | Administrative overview |
| `/about` | Project information |

---

## 🧪 Testing

The supplied project documentation describes automated tests covering model loading, database operations, prediction workflows, input validation, and application routes.

Run the documented test suite from the application directory:

```bash
python -m unittest tests/test_app.py
```

The supplied documentation reports 13 tests. Run the tests against the current codebase to confirm the actual result.

Recommended checks include:

- Model artifact loading.
- Input validation.
- Low-risk and high-risk prediction scenarios.
- Prediction database insertion and retrieval.
- Analytics and chart data generation.
- PDF report generation.
- Application route responses.
- Missing artifact and invalid-input handling.

---

## 📊 Model Performance

The following metrics are reported in the supplied **NovaCredit AI v2.1 documentation**. They should be confirmed against the exact model artifact and evaluation results in the deployed repository.

| Metric | Reported Result |
|---|---:|
| Algorithm | Regularized Random Forest |
| Test Accuracy | 83.20% |
| 5-Fold Stratified CV Accuracy | 83.00% ± 1.53% |
| Training Accuracy | 87.98% |
| Test ROC-AUC | 0.8820 |
| High-Risk Precision | 84.21% |
| High-Risk Recall | 78.80% |

### Model Selection

The supplied benchmark compares Logistic Regression, Decision Tree, Gradient Boosting, Random Forest, and HistGradientBoosting.

According to that documentation, the regularized Random Forest was selected as the production model based on its reported cross-validation and test results.

**Evaluation note:** Accuracy and other metrics depend on the dataset, target definition, split, and threshold. These figures describe the documented evaluation, not a guarantee of real-world lending performance.

---

## 🎯 Decision Threshold Analysis

A classification threshold changes the trade-off between identifying risky applicants and avoiding false alarms.

The supplied v2.1 documentation reports the following results:

| Default Probability Threshold | Test Accuracy | High-Risk Precision | High-Risk Recall |
|---:|---:|---:|---:|
| 0.30 | 76.10% | 68.33% | 91.01% |
| 0.40 | 82.60% | 79.13% | 85.22% |
| 0.45 | 83.60% | 82.58% | 82.23% |
| 0.50 | 83.20% | 84.21% | 78.80% |
| 0.60 | 81.90% | 89.07% | 69.81% |

A lower threshold identifies more potentially risky cases but may produce more false positives. A higher threshold can improve precision while missing more risky applicants.

Thresholds should be selected according to the lender's objectives, validation results, and applicable requirements.

---

## 🔮 Roadmap

Potential future improvements include:

- 🔐 User authentication and role-based access control.
- 🧠 SHAP-based model explanations.
- 🧪 Model versioning and systematic model comparisons.
- 📧 Email delivery for generated reports.
- 🐳 Docker-based deployment.
- 🗄️ PostgreSQL integration for durable multi-user storage.
- 📈 Model drift monitoring and performance tracking.
- 🛡️ Improved security, audit logging, and data governance.

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request with a clear description.

Please avoid committing credentials, private customer information, or other sensitive data.

---

## 📜 License

This project is documented as being distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file in the repository for the applicable terms.

---

### 💜 Made with Python, Machine Learning, and a passion for FinTech.

⭐ **If you find NovaCredit AI useful, consider giving the repository a star!**

**Developer:** [Ayush Srivastava](https://github.com/Ayush-4642)

