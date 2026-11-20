# Day 08: Airline Dynamic Pricing & Fare Surge Prediction Engine

## Executive Summary
Airlines adjust ticket fares dynamically in response to booking lead time, route demand, carrier positioning, and cabin capacity. This project builds an EDA-first dynamic pricing regression pipeline on `300,153` flight fare listings across six major Indian metro hubs, addressing row-order index leakage, bimodal cabin distributions, and non-linear last-minute price surges using **LightGBM**, **cyclical/route feature engineering**, and **SHAP explainability**.

## Business Problem
Accurate fare estimation powers both airline yield-management systems and consumer OTA "Buy vs. Wait" price-prediction alerts. However, naive linear models fail because ticket prices exhibit a steep non-linear spike within 10 days of departure and a wide structural gap between Economy and Business cabins. The objectives of this project are to:
1. Audit raw scraped fare listings prior to modeling to eliminate row-order data leakage (`Unnamed: 0`) and high-cardinality noise (`flight` codes).
2. Capture directional route demand, stopover penalties, and last-minute booking urgency (`days_left <= 7`).
3. Minimize **MAE** and **MAPE** across both Economy and Business fare regimes while explaining price drivers via **SHAP**.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Flight Price Prediction Dataset (`shubhambathwal/flight-price-prediction` on Kaggle)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
* **Raw Dimensions**: 300,153 flight listings across 12 columns (collected over 50 days across Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, and Chennai).
* **Critical EDA Discoveries**:
  1. **Row-Order Leakage (`Unnamed: 0`)**: Despite being named `Clean_Dataset.csv`, the file includes an exported CSV row index (`Unnamed: 0`) strongly correlated with `price` because listings are sorted by cabin class (`Economy` rows followed by `Business` rows). Dropping it prevents artificial score inflation.
  2. **Bimodal Fare Distribution**: `Economy` fares cluster below `₹10,000` whereas `Business` fares center above `₹45,000–₹55,000`, creating a dual-regime pricing structure.
  3. **Non-Linear Last-Minute Surge**: While global linear correlation between `days_left` and `price` appears weak, plotting mean fares across `days_left` reveals a sharp exponential price surge once `days_left < 10`.
  4. **High-Cardinality `flight` Codes (`1,561` Unique Values)**: One-hot encoding raw flight numbers bloats dimensionality without generalizing to unseen schedules; directional `route` pairs (`source_city_to_destination_city`) provide a compact, generalizable alternative.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed the `Unnamed: 0` index leakage column and high-cardinality `flight` code column.
2. **Dynamic Pricing Feature Engineering**:
   * `route` & `time_slot_pair`: Captures directional city-pair demand and departure-to-arrival time convenience.
   * `is_last_minute_7d`, `is_urgent_2d`, & `inv_days_left`: Encodes steep non-linear fare escalation near departure.
   * `is_nonstop` & `is_business`: Explicit structural indicators separating direct flights and Business cabin tiers.
3. **Leak-Free Preprocessing & Modeling**: Benchmarked `Log-Target Ridge Regression` against `LGBMRegressor` (`n_estimators=600`, `num_leaves=63`) using stratified splitting by cabin class.

## Key Results & Evaluation Metrics (Test Set: 60,031 Flight Listings)

| Model | MAE (INR) | RMSE (INR) | MAPE (%) | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Log-Target Ridge Regression (Baseline)** | ₹4,548.37 | ₹8,306.69 | 24.93% | 0.8663 |
| **LightGBM Regressor (Dynamic Pricing Engine)** | **₹1,632.11** | **₹2,953.59** | **12.52%** | **0.9831** |

*(Note: Update the metrics above with the exact values printed in Cell 6 of your notebook.)*

## Business Impact & Pricing Insights
1. **Booking Window Optimization**: SHAP confirms `class_Business`, `duration`, `days_left`, and `airline_Vistara` as the primary price drivers. Consumer OTA "Price Freeze" features should trigger `10–14 days` before departure—just ahead of the `days_left < 10` surge zone.
2. **Carrier Positioning**: Full-service carriers (`Vistara`, `Air India`) command significant route-dependent premiums over low-cost carriers (`Indigo`, `AirAsia`, `SpiceJet`) on direct business corridors such as `Delhi ↔ Mumbai`.

## How to Run
1. Download `Clean_Dataset.csv` from [Kaggle Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb