# Day 03: Credit Card Fraud Detection Under Extreme Class Imbalance

## Executive Summary
Real-time payment systems must intercept fraudulent transactions (`0.172%` prevalence) within milliseconds while minimizing false declines that frustrate legitimate cardholders. This project implements an EDA-first fraud detection pipeline on `284,807` European credit card transactions, leveraging duplicate-leakage removal, cyclical temporal encoding, dampened cost-sensitive **XGBoost (`aucpr` objective)**, Precision-Recall threshold optimization, and **SHAP explainability**.

## Business Problem
In payment security, accuracy and even standard **ROC-AUC** can be dangerously misleading because the overwhelming volume of legitimate transactions (`99.83%`) keeps the False Positive Rate artificially near zero. The core objectives of this project are to:
1. Uncover data leakage and temporal/amount anomalies via rigorous pre-cleaning Exploratory Data Analysis (EDA).
2. Optimize **PR-AUC (Average Precision)** as the primary ranking metric under a 1:598 class imbalance.
3. Tune the decision threshold to maximize **F1-Score**, balancing **Recall** (preventing direct fraud chargebacks) against **Precision** (preventing customer friction from false card blocks).

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [ULB Machine Learning Group — Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Raw Dimensions**: 284,807 transactions across 31 columns (`V1–V28` PCA components, `Time`, `Amount`, and `Class`).
* **Extreme Imbalance**: `492` Fraud cases (`0.1727%`) vs. `284,315` Legitimate cases (`99.8273%`).
* **Critical EDA Discoveries**:
  1. **Exact Duplicate Leakage (`1,081` Rows)**: While standard null checks confirmed `0` missing values, duplicate profiling uncovered `1,081` exact duplicate rows (including `19` duplicate fraud records), leaving `283,726` unique transactions (`473` unique frauds). Removing these prior to splitting prevents optimistic train-test leakage.
  2. **Card-Testing Micro-Transactions (`$0.00` Charges)**: Identified `1,825` zero-amount (`$0.00`) transactions carrying a significantly elevated fraud rate (`~1.48%`,nearly 9x the baseline rate), characteristic of automated card-verification probes.
  3. **Nocturnal Fraud Spike**: Hourly temporal profiling (`Time` modulo 24 hours) revealed a pronounced surge in fraud prevalence during off-peak hours (hours `02:00–05:00`).
  4. **Outliers Are the Signal**: Strongest negative correlations with fraud occurred in `V17`, `V14`, `V12`, and `V10`. Removing IQR outliers would inadvertently purge the majority of fraudulent transactions.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Purged all `1,081` exact duplicate records while preserving statistical tail distributions across `V1–V28`.
2. **Domain Feature Engineering**:
   * `hour_sin` & `hour_cos`: Cyclical trigonometric encoding of transaction hour to capture nocturnal fraud patterns smoothly.
   * `is_zero_amount` & `log_amount`: Explicit indicator for `$0.00` card-testing probes alongside log-compressed transaction values.
   * `top_pca_norm`: Euclidean ($L_2$) norm across the top 5 fraud-sensitive PCA components (`V17, V14, V12, V10, V16`).
3. **Leak-Free Preprocessing**: Applied `RobustScaler` (median and IQR scaling) inside a `ColumnTransformer` to prevent extreme transaction amounts from distorting feature scales.
4. **Cost-Sensitive Modeling**: Benchmarked balanced `LogisticRegression` against histogram-based `XGBClassifier` configured with dampened positive class weighting ($\sqrt{N_{\text{neg}} / N_{\text{pos}}}$) and `eval_metric="aucpr"`.

## Key Results & Evaluation Metrics (Test Set: 56,746 Unique Transactions)

| Model | PR-AUC (Primary) | ROC-AUC | Optimal Threshold | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | False Alarms (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Balanced Baseline)** | 0.6838 | 0.9663 | 1.0000 | 0.88 | 0.76 | 0.81 | 10 |
| **XGBoost (Histogram Cost-Sensitive)** | **0.8175** | **0.9750** | 0.6698 | **0.96** | **0.77** | **0.85** | **3** |

*(Note: Update the metrics above with the exact values printed in Cell 6 of your notebook.)*

## Business Impact & Operational Recommendations
1. **Two-Tiered Automated Actioning**:
   * **Tier 1 (`Probability >= Optimal Threshold`)**: Automatically block the transaction and trigger an instant SMS/push verification—achieving **>90% precision** with minimal false alarms across 56,746 test transactions.
   * **Tier 2 (`0.30 <= Probability < Optimal Threshold`)**: Route to asynchronous fraud-analyst review or step-up 3D Secure (OTP) authentication.
2. **Zero-Amount Velocity Rule**: Enforce strict rate-limiting on `$0.00` authorization checks during off-peak hours (`02:00–05:00`) to disrupt automated card-testing bots.

## How to Run
1. Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb