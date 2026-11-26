# Day 11: Bank Term Deposit Marketing Propensity & Leakage-Free Call Ranking

## Executive Summary
Retail banks rely on outbound telemarketing to market term deposit products, yet only `11.27%` of contacted clients subscribe. A well-known pitfall in this benchmark dataset is including `duration` (phone call length), which causes **100% post-outcome target leakage** since call duration is unknown before dialing. This project builds a strictly **deployable, pre-dial propensity pipeline** on `41,188` client interactions (`41,176` after deduplication), combining sentinel-value decomposition, macroeconomic and contact-fatigue features, **LightGBM ranking**, **Top-20% Call Capture Lift**, and **SHAP explainability**.

## Business Problem
Dialing every client wastes agent capacity and causes customer annoyance through repeated follow-up calls. The objectives of this project are to:
1. Conduct thorough pre-cleaning Exploratory Data Analysis (EDA) to expose post-call leakage (`duration`), sentinel numeric encodings (`pdays == 999`), and contact fatigue effects.
2. Build a realistic **pre-dial ranking model** evaluated by **PR-AUC**, **ROC-AUC**, and **Top-20% Decile Capture Rate**.
3. Provide clear campaign-policy recommendations derived from **SHAP** attributions.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Bank Marketing Dataset (`henriqueyamahata/bank-marketing` on Kaggle)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) (`bank-additional-full.csv`, semicolon-delimited).
* **Raw Dimensions**: 41,188 client interactions across 21 attributes (`12` exact duplicate rows removed).
* **Target Distribution**: `11.27%` Subscribed (`yes`) vs. `88.73%` Did Not Subscribe (`no`).
* **Critical EDA Discoveries**:
  1. **Post-Call Target Leakage (`duration`)**: Median call duration for subscribers is more than double that of non-subscribers, but `duration` is only observed *after* the call concludes. Dropping `duration` is mandatory for any production pre-dial scoring system.
  2. **Sentinel Encoding (`pdays == 999`)**: `96.32%` of records use `999` as a placeholder for *"never contacted previously"*. Passing `999` directly to linear scalers distorts distance and regression coefficients.
  3. **Diminishing Returns (`campaign`) & High-Intent Segments**: Conversion falls sharply as repeated calls exceed `3+` contacts, while `student` and `retired` clients as well as prior campaign successes (`poutcome == success`) convert at dramatically higher rates.
  4. **Macroeconomic Regime Sensitivity**: `euribor3m` (3-month Euribor interest rate) and `nr.employed` exhibit strong negative correlation with term deposit subscription rates.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed `12` exact duplicates, dropped `duration` (post-call leakage) and `default` (near-zero variance with only `3` positive cases), and retained `"unknown"` strings as explicit informative categories.
2. **Domain Feature Engineering**:
   * `was_previously_contacted` & `pdays_clean`: Decomposed the `999` sentinel into a clean binary indicator and valid elapsed days.
   * `is_high_intent_job`, `is_overcontacted` (`campaign >= 4`), & `is_peak_season_month`: Encodes demographic propensity, contact-fatigue cutoffs, and peak-month seasonality (`Mar, Sep, Oct, Dec`).
3. **Pre-Dial Modeling**: Benchmarked balanced `LogisticRegression` against cost-sensitive `LGBMClassifier` inside a leak-free `ColumnTransformer`.

## Key Results & Evaluation Metrics (Test Set: 8,236 Pre-Dial Prospects)

| Model | PR-AUC | ROC-AUC | Top-20% Call Capture | Optimal F1 Threshold | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Pre-Dial Baseline)** | 0.4441 | 0.7996 | 64.33% | 0.6555 | 0.45 | **0.59** | 0.51 | 0.87 |
| **LightGBM (Pre-Dial Propensity Engine)** | **0.4768** | **0.8092** | **66.70%** | 0.4594 | **0.50** | 0.56 | **0.52** | **0.89** |

## Business Impact & Telemarketing Policy Recommendations
1. **Cap Repeated Outreach at 3 Attempts**: Since conversion decays steadily beyond the 3rd contact (`is_overcontacted`), enforcing a strict 3-call cap reallocates agent hours to fresh, higher-propensity segments (`student`, `retired`, and `poutcome == success`).
2. **Macro-Aware Campaign Timing**: SHAP highlights `euribor3m`, `nr.employed`, `month`, and `poutcome_success` as dominant pre-dial drivers—marketing teams should concentrate outbound campaign budgets during low-Euribor windows and peak seasonal months (`Mar, Sep, Oct, Dec`).

## How to Run
1. Download `bank-additional-full.csv` from [Kaggle Bank Marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb