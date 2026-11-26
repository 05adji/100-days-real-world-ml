# Day 05: E-Commerce & Food Delivery ETA Prediction Engine

## Executive Summary
Accurate Estimated Time of Arrival (ETA) predictions are critical for on-demand food and last-mile delivery platforms. Under-predicting delivery times erodes customer trust and triggers support escalations, while over-predicting reduces checkout conversion. This project builds an EDA-first machine learning pipeline on `45,593` last-mile delivery logs to predict delivery duration in minutes using geospatial Haversine distance engineering, GPS sensor sanitization, **LightGBM regression**, and **SHAP explainability**.

## Business Problem
Real-world logistics telemetry suffers from GPS dropouts, string-encoded nulls, and complex interactions between road traffic, weather, and order batching (`multiple_deliveries`). The objectives of this project are to:
1. Audit raw logistics telemetry via pre-cleaning Exploratory Data Analysis (EDA) to isolate corrupted GPS coordinates and rating anomalies.
2. Minimize Mean Absolute Error (**MAE** in minutes) and maximize the **Within ±5 Minutes SLA Accuracy** rate.
3. Quantify the operational delay caused by traffic congestion, severe weather, and multi-order batching using **SHAP**.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Food Delivery Operations Analytics Dataset (Kaggle)](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset/versions/1)
* **Raw Dimensions**: 45,593 delivery records across 20 operational, geospatial, and environmental attributes.
* **Critical EDA Discoveries**:
  1. **Hidden String `"NaN"` Literals**: Standard null checks missed thousands of missing values stored as literal `"NaN "` strings with trailing whitespace across `Delivery_person_Age`, `Delivery_person_Ratings`, `Weather_conditions`, and `Road_traffic_density`.
  2. **Corrupted GPS Telemetry (`0.0, 0.0` & Inverted Negative Signs)**: Discovered `407` records with `(0.0, 0.0)` coordinates and `431` records with negative latitudes/longitudes in Indian cities (`8°N–37°N`, `68°E–97°E`). Computing raw Haversine distance without sanitizing these coordinates produces artificial 8,000+ km distances.
  3. **Out-of-Bounds Courier Ratings**: Identified `53` records where `Delivery_person_Ratings == 6.0` on a 1.0–5.0 rating scale.

## Methodology & Architecture
1. **Evidence-Based Telemetry Cleaning**: Stripped whitespace, converted `"NaN"` strings to `np.nan`, stripped the `"(min)"` prefix from target values, nullified ratings `> 5.0`, corrected sign-inverted Indian GPS coordinates via `abs()`, and masked `(0.0, 0.0)` GPS dropouts prior to distance calculation.
2. **Geospatial & Temporal Feature Engineering**:
   * `delivery_distance_km`: Great-circle Haversine distance between restaurant and customer coordinates.
   * `prep_time_mins`: Elapsed kitchen preparation time between `Time_Ordered` and `Time_Order_picked` (handling midnight wrap-around).
   * `is_dinner_rush` & `is_lunch_rush`: Peak-hour indicators (`18:00–22:00` and `11:00–14:00`).
   * `city_hub_code`: Extracted operational city prefix (e.g., `BANG`, `MUM`, `DEL`) from `Delivery_person_ID`.
3. **Leak-Free Preprocessing**: Combined median imputation + `StandardScaler` for numerical variables, ordered `OrdinalEncoder` (`Low < Medium < High < Jam`) for traffic density, and `OneHotEncoder` for nominal features inside a `ColumnTransformer`.

## Key Results & Evaluation Metrics (Test Set: 9,119 Deliveries)

| Model | MAE (Mins) | RMSE (Mins) | MAPE (%) | R² Score | Within ±5 Mins SLA (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression (Linear Baseline)** | 4.85 mins | 6.07 mins | 21.50% | 0.5792 | 58.45% |
| **LightGBM Regressor (ETA Engine)** | **3.21 mins** | **4.03 mins** | **13.96%** | **0.8145** | **80.00%** |

## Business Impact & Operational Recommendations
1. **Dynamic Batching Buffer (`multiple_deliveries`)**: SHAP analysis shows that stacking multiple deliveries on a single courier trip is one of the strongest positive drivers of delivery duration. Dispatch algorithms should restrict 3-order batching during `Jam` traffic states under `Stormy`/`Fog` weather.
2. **Vehicle Condition & Courier Age Interaction**: Poor vehicle condition (`Vehicle_condition == 0`) significantly amplifies travel time across distances `> 10 km`; long-distance orders should be prioritized for couriers with `Vehicle_condition >= 2`.

## How to Run
1. Download the dataset from [Kaggle Food Delivery Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset/versions/1) and place `train.csv` (or `Zomato Dataset.csv`) in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb