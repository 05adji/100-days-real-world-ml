# Day 02: Consumer Credit Risk & Loan Default Scoring Engine

## Executive Summary
Accurate credit underwriting requires distinguishing creditworthy borrowers from high-risk applicants while maintaining regulatory explainability. This project implements an EDA-first machine learning pipeline using the Kaggle Credit Risk Dataset (`32,581` loan records) to predict **Probability of Default (PD)** using **LightGBM**, **Kolmogorov-Smirnov (KS) Statistic**, **PR-AUC optimization**, and **SHAP explainability**.

## Business Problem
In consumer lending, **False Negatives** (approving a borrower who defaults) result in direct principal loss, whereas **False Positives** (rejecting a creditworthy borrower) result in lost interest revenue. The objective of this project is to:
1. Audit raw bureau and application data to eliminate corrupt records and prevent imputation data leakage.
2. Maximize rank-ordering power across imbalanced default classes (**PR-AUC**, **ROC-AUC**, and **KS-Statistic**).
3. Provide transparent, applicant-level risk attributions using **SHAP** to support underwriting policy rules.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Credit Risk Dataset (`laotse/credit-risk-dataset` on Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
* **Raw Dimensions**: 32,581 rows and 12 features.
* **Target Distribution**: ~21.82% Default (`1`) vs. ~78.18% Non-Default (`0`).
* **Critical EDA Discoveries**:
  1. **Exact Duplicates**: Identified and removed `165` exact duplicate rows (`0.51%`).
  2. **Biologically Impossible Outliers**: Uncovered severe data entry errors in `person_age` (up to **144 years old**) and `person_emp_length` (up to **123 years of employment**), which would distort linear scalers and split thresholds if left uncleaned.
  3. **Non-Random Missing Values**: Found `3,116` missing values (`9.56%`) in `loan_int_rate` and `895` missing values (`2.75%`) in `person_emp_length`. Preserved missingness indicator flags prior to leak-free median imputation inside the training pipeline.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed duplicate records and filtered out corrupted rows (`person_age > 100` or `person_emp_length > 60` / exceeding working-age bounds).
2. **Domain Feature Engineering**:
   * `log_person_income`: Log-transformed heavily right-skewed borrower income.
   * `computed_loan_to_income`: Cross-verified the exact ratio of `loan_amnt` to `person_income`.
   * `cred_hist_to_adult_age_ratio`: Ratio of credit history length to adult age (`person_age - 18`).
   * `is_int_rate_missing` & `is_emp_length_missing`: Explicit binary indicators capturing missing bureau attributes.
3. **Leak-Free Preprocessing**: Applied `OrdinalEncoder` (`A` through `G`) for `loan_grade`, `OneHotEncoder` for nominal features (`person_home_ownership`, `loan_intent`), and median imputation + `StandardScaler` for numerical features inside a `ColumnTransformer`.

## Key Results & Evaluation Metrics (Test Set: 6,482 Loan Applications)

| Model | ROC-AUC | PR-AUC | KS-Statistic | Optimal Threshold | Precision (Default) | Recall (Default) | F1-Score (Default) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 0.8785 | 0.7130 | 0.6064 | 0.6349 | 0.62 | 0.70 | 0.66 | 0.84 |
| **LightGBM (Credit Scoring GBDT)** | **0.9533** | **0.9131** | **0.7747** | 0.6596 | **0.95** | **0.77** | **0.85** | **0.94** |

## Business Impact & Underwriting Recommendations
1. **Debt-to-Income Cap (`loan_percent_income`)**: SHAP analysis confirms `loan_percent_income` and `loan_grade` as the dominant drivers of default. Applications where the loan exceeds **30% of annual income** for non-homeowners (`RENT`) exhibit a sharp spike in default risk and should trigger manual underwriting review.
2. **Tiered Pricing Verification**: Borrowers in grades `D–G` with prior default records (`cb_person_default_on_file == 1`) require stricter collateral or lower credit limits.

## How to Run
1. Download `credit_risk_dataset.csv` from [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) and place it in `data/raw/`.
2. Install dependencies and run the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb