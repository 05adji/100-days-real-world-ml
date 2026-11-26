# Day 10: Multi-Store Retail Sales & Holiday Demand Prediction

## Executive Summary
Accurate store-level weekly demand forecasting prevents costly stockouts during peak shopping holidays and reduces excess inventory holding costs. This project builds a strictly leak-free, EDA-first forecasting pipeline across `45` Walmart stores over `143` weeks, enforcing **Chronological Out-of-Time validation**, store-specific autoregressive lag/rolling features, **LightGBM regression**, **WAPE (Weighted Absolute Percentage Error)** evaluation, and **SHAP explainability**.

## Business Problem
In multi-store retail planning, random train-test splits leak future holiday spikes and macroeconomic shifts into training folds, producing unrealistically optimistic accuracy that fails in production. The objectives of this project are to:
1. Audit multi-store panel completeness, date formatting (`DD-MM-YYYY`), and holiday seasonality prior to feature engineering.
2. Enforce a strict **Chronological Out-of-Time split** (80% earliest weeks for training, 20% unseen future weeks for testing).
3. Minimize **WAPE (%)**, **MAE**, and **RMSE** while quantifying the impact of historical sales momentum versus macroeconomic indicators (`CPI`, `Unemployment`, `Fuel_Price`).

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Walmart Store Sales Dataset (`yasserh/walmart-dataset` on Kaggle)](https://www.kaggle.com/datasets/yasserh/walmart-dataset)
* **Raw Dimensions**: 6,435 weekly records (`45` stores × `143` consecutive weeks from `2010-02-05` to `2012-11-01`), `0` missing values, `0` duplicate rows.
* **Critical EDA Discoveries**:
  1. **Balanced Panel & `DD-MM-YYYY` Date Format**: Every store contains exactly `143` weekly observations; raw dates follow `DD-MM-YYYY` formatting (`dayfirst=True` required to avoid month/day inversion).
  2. **Extreme Store Volume Disparity**: Weekly store revenues span from `~$210k` to `~$3.82M`, confirming that `Store` IDs (`1–45`) represent distinct categorical fixed effects rather than continuous numeric magnitudes.
  3. **Year-End Holiday Surges**: Aggregate sales peak sharply around **Thanksgiving (Week 47)** and **Christmas (Weeks 51–52)**, whereas raw macroeconomic linear correlations (`Temperature`, `Fuel_Price`, `CPI`) remain weak globally.

## Methodology & Architecture
1. **Evidence-Based Cleaning & Ordering**: Parsed `Date` with `dayfirst=True`, sorted strictly by `['Store', 'Date']`, and cast `Store` to categorical strings.
2. **Leak-Free Temporal & Lag Feature Engineering**:
   * `sales_lag_1w`, `sales_lag_2w`, `sales_lag_4w`: Store-level lagged weekly sales (`shift(1..4)`).
   * `sales_rolling_4w_mean` & `sales_rolling_4w_std`: 4-week trailing demand level and volatility per store (`shift(1).rolling(4)`).
   * `week_of_year`, `month`, & `is_year_end_peak`: Captures recurring seasonal holiday surges.
3. **Out-of-Time Benchmarking**: Benchmarked `Ridge Regression` against `LGBMRegressor` on the final 20% unseen future weeks (`~1,260` store-weeks).

## Key Results & Evaluation Metrics (Out-of-Time Test Set)

| Model | WAPE (%) | MAE (USD) | RMSE (USD) | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression (Temporal Lag Baseline)** | 4.86% | $50,477.91 | $69,653.39 | 0.9829 |
| **LightGBM Regressor (Retail Demand Engine)** | **3.91%** | **$40,623.04** | **$58,074.60** | **0.9881** |

## Business Impact & Supply Chain Recommendations
1. **Inventory Replenishment Accuracy**: Achieving **<5% WAPE** on unseen future weeks enables automated store-level safety-stock replenishment based on trailing 4-week rolling momentum (`sales_rolling_4w_mean`) and `week_of_year`.
2. **Pre-Holiday Buffer Allocation**: Because year-end holiday weeks drive sharp non-linear surges, regional distribution centers should pre-stage buffer inventory 2 weeks ahead of Week 47 and Weeks 51–52.

## How to Run
1. Download `Walmart.csv` from [Kaggle Walmart Dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb