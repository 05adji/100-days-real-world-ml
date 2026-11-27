# Day 15: CRM Inbound Lead Scoring & Hot-Lead Prioritization Engine

## Executive Summary
Sales teams at high-volume EdTech and B2C platforms waste substantial calling capacity on low-intent web leads when only ~38.5% ultimately convert. Real-world CRM exports frequently suffer from hidden dropdown placeholders (`'Select'`), zero-variance form fields, and post-call sales tags that leak target outcomes. This project builds a strictly **deployable, pre-call lead scoring pipeline** on `9,240` leads across `37` attributes, combining `'Select'` sanitization, sales-leakage removal, engagement & profile-completeness engineering, **LightGBM scoring**, and **SHAP explainability**.

## Business Problem
To raise outbound sales conversion efficiency toward the company's ~80% target for prioritized leads, the CRM requires a `0–100` propensity score assigned immediately upon lead capture. The objectives of this project are to:
1. Audit raw CRM fields prior to modeling to expose hidden `'Select'` defaults, constant columns, and post-call sales leakage (`Tags`, `Lead Quality`).
2. Engineer web-engagement intensity and form-completeness signals from pre-call inbound telemetry.
3. Maximize **ROC-AUC**, **PR-AUC**, and **Top-30% Hot-Lead Conversion Rate** so sales reps focus on high-probability prospects.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Leads Dataset (`ashydv/leads-dataset` on Kaggle)](https://www.kaggle.com/datasets/ashydv/leads-dataset)
* **Raw Dimensions**: 9,240 inbound leads across 37 columns (`0` duplicate rows).
* **Target Distribution**: `38.54%` Converted (`1`) vs. `61.46%` Not Converted (`0`).
* **Critical EDA Discoveries**:
  1. **Hidden `'Select'` Placeholders**: Multiple web-form dropdowns (`Specialization`, `Lead Profile`, `City`, `How did you hear about X Education`) store `'Select'` when left blank by users, masking true missingness from `isnull().sum()`.
  2. **Post-Call Sales Leakage (`Tags`, `Lead Quality`, `Asymmetrique*`)**: Sales-assigned tags (`'Closed by Horizzon'`, `'Ringing'`, `'Will revert'`) and internal call-quality ratings are populated *after* agent outreach; removing them prevents unrealistic offline inflation.
  3. **Zero-Variance & Case-Mismatch Noise**: Identified multiple single-value constant columns (`Magazine`, `Update me on Supply Chain Content`, etc.) and case-duplicated `Lead Source` categories (`'Google'` vs. `'google'`).
  4. **Bot/Crawler Outliers**: `TotalVisits` reaches `251` while its 99th percentile sits far lower, requiring leak-free percentile capping inside `X_train`.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Replaced `'Select'` with `np.nan`, removed zero-variance constants and post-call leakage columns, unified `'google'` into `'Google'`, and imputed missing profile fields as `'Unknown'`.
2. **Inbound Feature Engineering**:
   * `avg_time_per_visit` & `is_zero_website_engagement`: Quantifies browsing depth per visit.
   * `form_fields_completed`: Counts optional form fields filled (`0–4`) as an intent proxy.
   * `is_working_professional`: Flags the highest-converting occupation cohort discovered in EDA.
3. **Leak-Free Preprocessing & Benchmarking**: Applied 99th-percentile outlier capping and rare-category grouping (`< 1.5%`) strictly on `X_train`, benchmarking `LogisticRegression` against `LGBMClassifier`.

## Key Results & Evaluation Metrics (Test Set: 1,848 Inbound Leads)

| Model | ROC-AUC | PR-AUC | Top-30% Hot-Lead Conversion | Optimal F1 Threshold | Precision (Converted) | Recall (Converted) | F1-Score (Converted) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Inbound Baseline)** | 0.8902 | 0.8304 | 81.59% | 0.3625 | 0.68 | **0.90** | 0.78 | 0.80 |
| **LightGBM (Inbound Engine)** | **0.8997** | **0.8550** | **83.21%** | 0.3200 | **0.71** | 0.89 | **0.79** | **0.82** |

### 10-Decile Lead Score Calibration Highlights (LightGBM)
* **Tier 1 — Hot Leads (Deciles 1–2, `Score >= 81.8`)**: `96.22%` and `83.78%` conversion rates, capturing **46.77%** of all sales from just 20% of leads.
* **Tier 2 — Priority Queue (Top 30%, `Score >= 65.6`)**: Averages **83.21% conversion** (exceeding the 80% target) while capturing **64.75%** of total conversions.
* **Tier 3 — Automated Nurturing Cutoff (Deciles 6–10, `Score < 29.0`)**: Bottom 50% of leads account for only **9.83%** of conversions, allowing sales to focus calling capacity exclusively on the top 50% (**90.17%** cumulative capture).

## Business Impact & Sales Operations Recommendations
1. **Exceeding the 80% Hot-Lead Conversion Target**: By routing only the **Top 30% highest-scored leads** to immediate phone outreach, the sales team achieves **>80% conversion** on prioritized calls without relying on post-call leakage tags.
2. **Lead-Magnet & Form Optimization**: SHAP highlights `Total Time Spent on Website`, `Lead Origin_Lead Add Form`, `Last Activity`, and `is_working_professional` as primary drivers; leads with `form_fields_completed >= 3` and high dwell time should trigger instant sales callbacks within 5 minutes.

## How to Run
1. Download `Leads.csv` from [Kaggle Leads Dataset](https://www.kaggle.com/datasets/ashydv/leads-dataset) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb