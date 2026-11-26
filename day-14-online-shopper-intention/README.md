# Day 14: E-Commerce Clickstream Purchase Intention Prediction

## Executive Summary
Most e-commerce browsing sessions end without a checkout (`84.53%`), making blanket promotional pop-ups margin-dilutive. This project builds an EDA-first clickstream conversion pipeline on `12,330` user sessions (`12,205` unique sessions after deduplication), addressing Google Analytics ID encoding, `BounceRates–ExitRates` multicollinearity, zero-inflated `PageValues`, **LightGBM classification**, and **SHAP explainability**.

## Business Problem
Real-time conversion-propensity scoring enables e-commerce platforms to trigger targeted incentives (e.g., free-shipping nudges or live-chat assistance) selectively to high-intent shoppers. The objectives of this project are to:
1. Audit raw Google Analytics session telemetry prior to cleaning to remove exact bounce-session duplicates and flag false-numeric IDs.
2. Engineer session-depth, dwell-time, and friction features while preventing rare-category overfitting.
3. Optimize **PR-AUC**, **ROC-AUC**, and **F1-Score** for imbalanced session conversion prediction.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Online Shoppers Purchasing Intention Dataset (`imakash3011/online-shoppers-purchasing-intention-dataset` on Kaggle)](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
* **Raw Dimensions**: 12,330 sessions across 18 behavioral and contextual attributes (`0` missing values).
* **Target Distribution**: `15.47%` Purchased (`1,908` sessions) vs. `84.53%` No Purchase (`10,422` sessions).
* **Critical EDA Discoveries**:
  1. **Duplicate Session Records (`125` Rows)**: Identified `125` exact duplicate rows (predominantly zero-duration bounce sessions), leaving `12,205` unique sessions.
  2. **Zero-Inflated `PageValues` Threshold**: Over `77%` of sessions record `PageValues == 0.0` (converting at `< 4%`), whereas sessions with `PageValues > 0` convert at a dramatically higher rate.
  3. **Severe Collinearity (`BounceRates` vs. `ExitRates`)**: Measured a Pearson correlation of `r = 0.91` between `BounceRates` and `ExitRates`.
  4. **False-Numeric GA Identifiers**: `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are stored as integers (`1–20`) despite representing nominal categories; grouping rare values (`< 1%`) inside `X_train` prevents high-dimensional noise.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed `125` duplicate rows, cast boolean flags (`Weekend`, `Revenue`) to integers, standardized `'June'` to `'Jun'`, and converted GA numeric IDs to categorical strings.
2. **Clickstream Feature Engineering**:
   * `total_pages`, `total_duration`, `avg_time_per_product_page`, & `product_page_share`: Captures session browsing depth and product-focused engagement.
   * `has_nonzero_page_value` & `log_page_values`: Explicitly separates zero-value browsing from commercial-intent sessions.
   * `exit_minus_bounce`: Isolates multi-page exit friction from single-page bounces.
3. **Modeling & Thresholding**: Benchmarked balanced `LogisticRegression` against cost-sensitive `LGBMClassifier` with F1-optimal thresholding.

## Key Results & Evaluation Metrics (Test Set: 2,441 Unique Sessions)

| Model | PR-AUC | ROC-AUC | Optimal F1 Threshold | Precision (Purchased) | Recall (Purchased) | F1-Score (Purchased) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | 0.6892 | 0.9270 | 0.6320 | 0.60 | **0.79** | 0.68 | 0.88 |
| **LightGBM (Clickstream Engine)** | **0.7407** | **0.9308** | 0.4977 | **0.66** | 0.74 | **0.70** | **0.90** |

### Ablation Study: Impact of `PageValues` (Commercial Funnel Signal)

| Feature Set | PR-AUC | ROC-AUC |
| :--- | :--- | :--- |
| **Full Clickstream (WITH `PageValues`)** | **0.7407** | **0.9308** |
| **Browsing-Only (WITHOUT `PageValues`)** | 0.3659 | 0.7862 |

## Business Impact & E-Commerce Recommendations
1. **Intent-Triggered Cart Nudges**: SHAP identifies `PageValues`, `Month` (November peak season), `ExitRates`, and `ProductRelated_Duration` as top conversion drivers; triggering dynamic checkout reminders only for sessions with `PageValues > 0` and low `ExitRates` protects promotional margins.
2. **New-Visitor Onboarding**: `New_Visitor` sessions convert at a higher rate per visit than `Returning_Visitor` sessions in several categories—streamlining guest checkout reduces friction for first-time buyers.

## How to Run
1. Download `online_shoppers_intention.csv` from [Kaggle Online Shoppers Purchasing Intention](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb