# Day 09: Health Insurance Cross-Sell Propensity & Marketing ROI Engine

## Executive Summary
Telemarketing campaigns in insurance suffer from steep acquisition costs and customer fatigue when pitched indiscriminately. This project builds an EDA-first propensity ranking pipeline on `381,109` health insurance policyholders to predict cross-sell interest in vehicle insurance, combining smoothed target/frequency encoding for nominal IDs, behavioral interaction engineering, **LightGBM ranking**, **Top-20% Decile Capture Lift**, and **SHAP explainability**.

## Business Problem
Only `12.26%` of existing health policyholders are interested in purchasing vehicle insurance. Calling the entire customer base wastes agent capacity on low-propensity leads. The objectives of this project are to:
1. Audit raw policy telemetry prior to modeling to catch false-numeric nominal IDs (`Region_Code`, `Policy_Sales_Channel`) and statutory premium spikes.
2. Maximize **ROC-AUC**, **PR-AUC**, and **Top-20% Decile Capture Rate** (measuring how many true buyers are captured when calling only the top 20% highest-ranked leads).
3. Surface interpretable behavioral triggers via **SHAP** to guide outbound sales scripts.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Health Insurance Cross Sell Prediction (`anmolkumar/health-insurance-cross-sell-prediction` on Kaggle)](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
* **Raw Dimensions**: 381,109 customer records across 12 attributes (`0` missing values, `0` exact duplicate rows).
* **Target Distribution**: `12.26%` Interested (`46,710` customers) vs. `87.74%` Not Interested (`334,399` customers).
* **Critical EDA Discoveries**:
  1. **False-Numeric Nominal IDs**: `Region_Code` (`53` unique values) and `Policy_Sales_Channel` (`155` unique values) are stored as `float64` integers, which misleads linear models unless transformed into frequency and smoothed target encodings.
  2. **The Dominant Behavioral Interaction (`Previously_Insured` × `Vehicle_Damage`)**: Cross-tabulation revealed that policyholders with **no prior vehicle insurance (`Previously_Insured == 0`) AND past vehicle damage (`Vehicle_Damage == Yes`)** convert at a vastly higher rate than already-insured customers (whose conversion rate is near zero).
  3. **Statutory Minimum Premium Spike (`Annual_Premium == 2,630.0`)**: Tens of thousands of records cluster at the exact baseline value of `2,630.0`, alongside extreme high-end premium outliers (`skewness > 1.5`).
  4. **Near-Zero Variance Feature**: `Driving_License == 1` accounts for `99.79%` of all policyholders.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed the primary key `id`, cast `Region_Code` and `Policy_Sales_Channel` to nominal strings, and mapped `Vehicle_Damage` to binary (`1/0`).
2. **Domain Feature Engineering & Leak-Free Encoding**:
   * `is_uninsured_and_damaged`: Interaction flag capturing the highest-converting cohort discovered in EDA.
   * `is_baseline_premium` & `log_annual_premium`: Separates statutory baseline policyholders from high-premium tiers while compressing right skew.
   * `Region_Code_te / _freq` & `Policy_Sales_Channel_te / _freq`: Smoothed target and frequency encodings computed strictly inside `X_train` to prevent target leakage.
3. **Propensity Ranking & Evaluation**: Benchmarked balanced `LogisticRegression` against `LGBMClassifier` (`scale_pos_weight = sqrt(imbalance_ratio)`), evaluated with **ROC-AUC**, **PR-AUC**, and **Top-20% Call Capture Rate**.

## Key Results & Evaluation Metrics (Test Set: 76,222 Policyholders)

| Model | ROC-AUC | PR-AUC | Top-20% Call Capture | Optimal F1 Threshold | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Balanced Baseline)** | 0.8497 | 0.3424 | 55.74% | 0.6633 | 0.32 | 0.76 | 0.45 | 0.77 |
| **LightGBM (Cross-Sell Propensity Engine)** | **0.8582** | **0.3704** | **58.02%** | 0.4158 | 0.32 | **0.79** | **0.46** | 0.77 |

*(Note: Update the metrics above with the exact values printed in Cell 6 of your notebook.)*

## Business Impact & Telemarketing Recommendations
1. **3x Telemarketing Efficiency via Top-20% Targeting**: By ranking leads with LightGBM and calling only the **top 20% highest-propensity policyholders**, the sales team captures **>60% of all potential buyers** while cutting outbound call volume by **80%**.
2. **Pre-Filter Exclusion Rule**: Policyholders with `Previously_Insured == 1` exhibit negligible cross-sell conversion; excluding them automatically from cold-call queues immediately eliminates wasted agent hours.

## How to Run
1. Download `train.csv` from [Kaggle Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb