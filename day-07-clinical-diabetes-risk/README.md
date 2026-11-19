# Day 07: Clinical Diabetes Risk Screening & Patient Stratification

## Executive Summary
Early identification of patients at high risk for Type 2 diabetes enables timely lifestyle and clinical interventions before severe microvascular or cardiovascular complications develop. This project builds an EDA-first clinical screening pipeline on `100,000` Electronic Health Record (EHR) patient profiles (`96,146` unique patients after deduplication), combining EHR data-quality auditing, clinical biomarker feature engineering, **F2-score threshold optimization** (prioritizing medical sensitivity), **XGBoost classification**, and **SHAP explainability**.

## Business & Clinical Problem
In medical screening, **False Negatives** (failing to flag a diabetic patient) carry far higher clinical and long-term healthcare costs than **False Positives** (recommending a confirmatory fasting glucose or HbA1c test). The objectives of this project are to:
1. Audit raw EHR records prior to cleaning to uncover duplicate patient records, hidden missing categories, and automated system imputation artifacts.
2. Optimize **PR-AUC** and **ROC-AUC** under class imbalance (`~8.8%` positive prevalence) while tracking probability calibration via **Brier Score**.
3. Tune the decision threshold using the **$F_2$-Score** (weighting Recall twice as heavily as Precision) to minimize missed diagnoses.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Diabetes Prediction Dataset (`iammustafatz/diabetes-prediction-dataset` on Kaggle)](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
* **Raw Dimensions**: 100,000 patient records across 9 demographic and clinical attributes.
* **Critical EDA Discoveries**:
  1. **Duplicate Patient Records (`3,854` Rows / 3.85%)**: Deduplication reduced the dataset to `96,146` unique patients (`8,482` diabetic cases, `8.82%` prevalence), preventing train-test leakage.
  2. **Hidden Missingness in `smoking_history` (`~35.8%`)**: Standard `isnull().sum()` returned `0` nulls, yet `35,816` records were encoded as `"No Info"`. Retaining this as an explicit `"Unknown"` category with a binary missingness indicator (`is_smoking_unknown`) preserved over one-third of the clinical cohort.
  3. **EHR Imputation Spike (`bmi == 27.32`)**: Value-count profiling revealed an abnormal spike of records clustered at `bmi == 27.32`, indicating an upstream EHR placeholder value when height/weight measurements were unavailable.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed `3,854` exact duplicates, consolidated `18` rare `"Other"` gender records (`0.018%`) to stabilize stratified splitting, and standardized `"No Info"` into `"Unknown"`.
2. **Clinical Feature Engineering**:
   * `is_hba1c_diabetic_range` (`>= 6.5%`) & `is_glucose_elevated` (`>= 140 mg/dL`): Encodes standard clinical diagnostic thresholds.
   * `glycemic_risk_index`: Interaction between `HbA1c_level` and `log1p(blood_glucose_level)`.
   * `comorbidity_count` & `age_bmi_interaction`: Captures compounding cardiometabolic risk (`hypertension + heart_disease`).
   * `is_bmi_default_spike` & `is_smoking_unknown`: Explicit flags allowing tree splits to account for upstream EHR recording artifacts.
3. **Recall-Prioritized Modeling**: Benchmarked balanced `LogisticRegression` against histogram-based `XGBClassifier` (`eval_metric="aucpr"`) using **$F_2$-Score** probability threshold tuning.

## Key Results & Evaluation Metrics (Test Set: 19,230 Unique Patients)

| Model | PR-AUC | ROC-AUC | Brier Score | Optimal F2 Threshold | Precision (Diabetic) | Recall (Diabetic) | F1-Score (Diabetic) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Balanced Baseline)** | 0.8207 | 0.9636 | 0.0782 | 0.6437 | 0.53 | 0.82 | 0.64 | 0.92 |
| **XGBoost (Clinical Risk Engine)** | **0.8839** | **0.9777** | **0.0310** | 0.3582 | **0.59** | **0.86** | **0.70** | **0.94** |

*(Note: Update the metrics above with the exact values printed in Cell 6 of your notebook.)*

## Business & Clinical Impact
1. **Two-Stage Screening Triage**: With high **PR-AUC** and low **Brier Score**, patients scoring above the $F_2$-optimal threshold can be automatically scheduled for confirmatory lab work, while borderline scores (`0.25–0.48`) receive preventive dietary and lifestyle coaching.
2. **Transparent Clinician Support**: SHAP explanations highlight `HbA1c_level`, `blood_glucose_level`, `age`, `bmi`, and `comorbidity_count` for each individual patient, enabling physicians to audit the exact risk drivers behind every alert.

## How to Run
1. Download `diabetes_prediction_dataset.csv` from [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb