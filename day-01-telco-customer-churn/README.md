# Day 01: Telco Customer Churn Prediction & Retention Analytics

## Executive Summary
Customer churn directly erodes recurring revenue in subscription-based telecommunications businesses. This project implements an end-to-end, EDA-first machine learning pipeline in Jupyter Notebook to identify high-risk subscribers before cancellation, combining raw data profiling, evidence-based cleaning, domain feature engineering, probability threshold optimization, and SHAP explainability.

## Business Problem
Acquiring a new telecommunications subscriber costs significantly more than retaining an existing one. However, blanket retention discounts waste marketing budget on customers who would have stayed anyway. The objective of this project is to:
1. Perform rigorous Exploratory Data Analysis (EDA) prior to data cleaning to uncover hidden data anomalies and high-risk customer segments.
2. Rank customers accurately by their churn probability (**ROC-AUC** & **PR-AUC**).
3. Optimize decision thresholds to balance **Recall** (capturing actual churners) and **Precision** (minimizing wasted retention incentives).
4. Explain model predictions using **SHAP (SHapley Additive exPlanations)** to guide actionable retention campaigns.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Dimensions**: 7,043 customer records across 21 attributes (`0` duplicate rows).
* **Target Distribution**: 26.54% Churned (`1,869` customers) vs. 73.46% Retained (`5,174` customers).
* **Critical EDA Discoveries**:
  1. **Hidden Missing Values in `TotalCharges`**: Standard null checks (`isnull().sum()`) reported `0` missing values across all columns. However, schema and whitespace profiling revealed **11 blank string records (`" "`)** in `TotalCharges` where `tenure == 0` (brand-new subscribers who had not completed their first billing cycle).
  2. **Redundant Service Labels**: Six add-on columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) contained `"No internet service"`, which is collinear with `InternetService == "No"`.
  3. **High-Risk Behavioral Segment**: Subscribers on **Month-to-month contracts** paying via **Electronic check** exhibited the highest concentration of early-tenure churn.

## Methodology & Architecture
1. **Evidence-Based Data Cleaning**:
   * Coerced `TotalCharges` to float and imputed `0.0` for the 11 `tenure == 0` records.
   * Consolidated redundant `"No internet service"` and `"No phone service"` categories into `"No"`.
   * Cast binary `SeniorCitizen` (`0/1`) to categorical strings for uniform encoding.
2. **Domain Feature Engineering**:
   * `charge_to_historical_ratio`: Ratio of current `MonthlyCharges` to historical average monthly spend to detect sudden bill shocks.
   * `total_addons`: Total count of active sticky value-added services per customer.
   * `is_high_risk_segment`: Interaction flag combining Month-to-month contracts with Electronic check payments.
3. **Leak-Free Preprocessing Pipeline**:
   * Numerical features imputed via median and standardized with `StandardScaler`.
   * Categorical features imputed via mode and encoded with `OneHotEncoder(handle_unknown="ignore")`.

## Key Results & Evaluation Metrics (Test Set: 1,409 Customers)

| Model | ROC-AUC | PR-AUC | Optimal Threshold | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | **0.8419** | 0.6350 | 0.5683 | 0.54 | **0.74** | 0.63 | 0.76 |
| **XGBoost (Tuned Class Weight)** | 0.8414 | **0.6615** | 0.6125 | **0.59** | 0.69 | 0.63 | **0.79** |

### Model Selection Trade-off Analysis
* **High-Recall Strategy (Logistic Regression)**: Captures **74% of all churning customers** (`ROC-AUC: 0.8419`), ideal when retention outreach costs (e.g., automated email/app offers) are low compared to customer lifetime value (LTV).
* **High-Precision Strategy (XGBoost)**: Achieves superior **PR-AUC (`0.6615`)** and **Precision (`0.59`)** with **79% overall accuracy**, making it preferable when retention incentives involve costly monetary discounts.

## Business Impact & Actionable Recommendations
1. **Contract Migration Incentive**: Target Month-to-month subscribers in their first 1–6 months with a discounted 1-year contract upgrade to lock in tenure.
2. **Payment Method Auto-Pay Nudge**: Migrate Electronic check users to automatic credit card or bank transfer billing via a one-time bill credit.
3. **Sticky Add-On Bundling**: Provide a complimentary 3-month trial of `TechSupport` and `OnlineSecurity` for Fiber Optic users experiencing high monthly charges.

## How to Run
1. Download the dataset from [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside `data/raw/`.
2. Install dependencies and launch the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook notebook.ipynb