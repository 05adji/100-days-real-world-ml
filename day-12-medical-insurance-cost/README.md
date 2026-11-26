# Day 12: Medical Insurance Cost Prediction & Actuarial Risk Modeling

## Executive Summary
Accurate individual healthcare cost estimation is essential for actuarial underwriting and transparent insurance premium pricing. Naive linear models trained on raw demographic features fail due to a sharp non-linear cost cliff when smoking coincides with clinical obesity (`BMI >= 30`). This project implements an EDA-first regression pipeline on `1,338` beneficiary records (`1,337` unique profiles), pairing **actuarial interaction engineering (`smoker_x_obese`, `smoker_x_bmi`)**, **5-fold cross-validated small-sample regularization**, **XGBoost vs. Ridge benchmarking**, and **SHAP explainability**.

## Business Problem
Underpricing high-risk health policyholders erodes insurer solvency, whereas overpricing low-risk individuals drives customer churn to competitors. The objectives of this project are to:
1. Uncover hidden data duplicates and non-linear cost regimes via pre-cleaning Exploratory Data Analysis (EDA).
2. Engineer clinically grounded interaction features so even interpretable linear baselines capture cost discontinuities.
3. Prevent overfitting on a compact (`n = 1,337`) dataset via **5-fold Cross-Validation**, shallow tree depth (`max_depth=3`), and $L_2$ regularization.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Medical Cost Personal Datasets (`mirichoi0218/insurance` on Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
* **Raw Dimensions**: 1,338 beneficiary records across 7 attributes (`0` missing values).
* **Critical EDA Discoveries**:
  1. **Exact Duplicate Record**: Detected `1` exact duplicate row (`19-year-old male, bmi 30.59, $1,639.56`), removed prior to train-test splitting to ensure strict evaluation integrity.
  2. **The `Smoker × Obese` Cost Cliff**: Smokers average `~$32,050` in annual charges versus `~$8,434` for non-smokers. Scatter profiling revealed a sharp structural jump at `BMI >= 30.0`, pushing obese smokers into the `$34,000–$63,770` regime.
  3. **Right-Skewed High-Cost Tail**: Medical charges span `$1,121.87` to `$63,770.43` (`skewness = 1.52`); high-end IQR outliers represent genuine high-risk clinical cases and were intentionally preserved.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed the `1` exact duplicate record and encoded `is_smoker` and `is_male` cleanly.
2. **Actuarial Feature Engineering**:
   * `is_obese` (`bmi >= 30.0`) & `smoker_x_obese`: Captures the discrete actuarial risk jump identified in EDA.
   * `smoker_x_bmi`: Models continuous BMI slope escalation among active smokers.
   * `age_squared` & `has_children`: Captures non-linear aging healthcare cost acceleration and family coverage tiers.
3. **Small-Data Regularized Benchmarking**: Benchmarked `Ridge(alpha=5.0)` (equipped with interaction terms) against a depth-regularized `XGBRegressor(max_depth=3, reg_lambda=5.0)` using 5-fold cross-validation and a stratified 80/20 holdout test set.

## Key Results & Evaluation Metrics (Test Set: 268 Beneficiaries)

| Model | 5-Fold CV RMSE | Test RMSE (USD) | Test MAE (USD) | MAPE (%) | R² Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression (With Interactions)** | **$4,636.11** | **$3,346.81** | **$1,990.68** | **28.20%** | **0.9223** |
| **XGBoost (Depth-Regularized)** | $4,744.45 | $3,361.15 | $2,087.21 | 30.71% | 0.9216 |

## Business & Actuarial Insights
1. **Interaction Engineering Bridges the Linear Gap**: Adding `smoker_x_obese` and `smoker_x_bmi` allows even a transparent `Ridge` model to rival gradient boosted trees—giving actuaries a fully auditable linear pricing formula when regulatory explainability is required.
2. **Preventive Wellness Incentives**: SHAP highlights `smoker_x_obese`, `is_smoker`, `age`, and `bmi` as the dominant cost drivers; subsidizing smoking-cessation and weight-management programs for the `BMI >= 30` smoker cohort directly targets the largest claim variance.

## How to Run
1. Download `insurance.csv` from [Kaggle Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb