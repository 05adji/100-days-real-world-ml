# Day 13: Employee Attrition Prediction & People Analytics Engine

## Executive Summary
Voluntary employee turnover disrupts team continuity and incurs steep replacement costs across recruiting and onboarding. Because HR tabular datasets are typically compact (`n = 1,470`) and imbalanced (`16.12%` attrition), unregularized tree models easily overfit. This project builds an EDA-first attrition prediction pipeline combining zero-variance column auditing, HR burnout/compensation feature engineering, **5-fold Stratified Cross-Validation**, **$L_2$-regularized Logistic Regression vs. depth-regularized XGBoost benchmarking**, and **SHAP explainability**.

## Business Problem
HR leadership needs an early-warning scoring system to identify flight-risk talent before resignation notices are filed, alongside transparent drivers to design retention policies. The objectives of this project are to:
1. Audit raw HR records prior to cleaning to remove zero-variance constants and identifier columns.
2. Engineer domain-grounded burnout, promotion-stagnation, and compensation-fairness indicators.
3. Evaluate generalization strictly via **5-Fold Stratified Cross-Validation** (`ROC-AUC` & `PR-AUC`) alongside holdout test metrics.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [IBM HR Analytics Employee Attrition & Performance (`pavansubhasht/ibm-hr-analytics-attrition-dataset` on Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* **Raw Dimensions**: 1,470 employee records across 35 attributes (`0` missing values, `0` duplicate rows).
* **Target Distribution**: `16.12%` Attrited (`237` employees) vs. `83.88%` Retained (`1,233` employees).
* **Critical EDA Discoveries**:
  1. **Zero-Variance Constant Columns**: Identified three completely constant columns (`EmployeeCount = 1`, `StandardHours = 80`, `Over18 = 'Y'`) and one unique primary key (`EmployeeNumber`), all flagged for removal.
  2. **Overtime Burnout Cliff**: Employees working `OverTime == Yes` exhibit roughly 3x the attrition rate of non-overtime peers.
  3. **Role & Equity Vulnerability**: `Sales Representative` roles experience the highest turnover (`~39.8%`), while employees with `StockOptionLevel == 0` churn at nearly triple the rate of vested peers (`StockOptionLevel >= 1`).

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Dropped `EmployeeCount`, `StandardHours`, `Over18`, and `EmployeeNumber`, and mapped `Attrition` and `OverTime` to binary integers.
2. **HR Domain Feature Engineering**:
   * `income_per_job_level`: Measures compensation competitiveness relative to organizational seniority.
   * `overtime_commute_stress`: Captures compounding burnout (`OverTime × DistanceFromHome`).
   * `years_without_promotion_ratio` & `tenure_stability_ratio`: Quantifies career stagnation and manager stability.
   * `overall_satisfaction`: Composite index across environment, job, relationship satisfaction, and work-life balance.
   * `is_early_career_unvested`: Flags younger employees (`Age <= 30`) lacking equity retention (`StockOptionLevel == 0`).
3. **Small-Sample Regularized Benchmarking**: Compared strong $L_2$-regularized `LogisticRegression(C=0.2)` against shallow `XGBClassifier(max_depth=3, reg_lambda=5.0)` using 5-fold Stratified CV.

## Key Results & Evaluation Metrics (Test Set: 294 Employees)

| Model | 5-Fold CV ROC-AUC | 5-Fold CV PR-AUC | Test ROC-AUC | Test PR-AUC | Precision (Attrition) | Recall (Attrition) | F1-Score (Attrition) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (L2 Regularized)** | **0.8342** | **0.6355** | **0.8052** | **0.5812** | **0.55** | 0.60 | **0.57** | **0.86** |
| **XGBoost (Depth-Regularized)** | 0.8227 | 0.6170 | 0.7623 | 0.4945 | 0.42 | **0.62** | 0.50 | 0.80 |

## Business Impact & HR Retention Recommendations
1. **Overtime & Commute Intervention**: SHAP highlights `OverTime`, `overtime_commute_stress`, `MonthlyIncome`, and `StockOptionLevel` as leading attrition triggers; HR should prioritize hybrid/remote flexibility for high-overtime, long-commute employees.
2. **Early-Career Retention Grants**: Extending modest equity/retention bonuses (`StockOptionLevel >= 1`) or structured promotion paths to high-performing `Age <= 30` staff directly mitigates the highest-turnover cohort.

## How to Run
1. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from [Kaggle IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb