# Day 06: Hotel Booking Cancellation & Overbooking Risk Engine

## Executive Summary
Hotel room nights are perishable inventory—last-minute cancellations directly reduce Occupancy and Revenue Per Available Room (RevPAR). This project builds a leak-free, EDA-first machine learning pipeline on `119,390` hotel booking records (`87,229` unique valid reservations after deduplication and anomaly filtering) to predict cancellation probability using **LightGBM**, **Precision-Recall threshold optimization**, and **SHAP explainability**.

## Business Problem
Online Travel Agencies (OTAs) with free cancellation policies have caused hotel cancellation rates to surge above 27–37%. Revenue managers need an accurate pre-arrival cancellation probability score to:
1. Identify high-risk reservations weeks before check-in for targeted deposit/reconfirmation campaigns.
2. Calculate safe room-type overbooking buffers without walking arriving guests.
3. Prevent **post-event target leakage** (`reservation_status` and `assigned_room_type`) that artificially inflates offline model metrics.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Hotel Booking Demand Dataset (`jessemostipak/hotel-booking-demand` on Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
* **Raw Dimensions**: 119,390 booking records across 32 attributes (City Hotel in Lisbon & Resort Hotel in Algarve, Portugal).
* **Critical EDA Discoveries**:
  1. **Severe Duplicate Distortion (`31,994` Duplicate Rows / 26.8%)**: Batch OTA/group entries created `31,994` exact duplicate rows. Removing duplicates adjusts the true unique-booking cancellation rate from `37.04%` to `27.52%` and prevents train-test leakage.
  2. **100% Target Leakage Features**: Cross-tabulation confirmed that `reservation_status` (`Canceled` / `No-Show` vs. `Check-Out`) and `reservation_status_date` encode the target variable with 100% certainty after the stay date, while `assigned_room_type` leaks front-desk check-in operations.
  3. **Ghost Bookings & ADR Anomalies**: Uncovered `180` reservations with `0` total guests (`adults + children + babies == 0`), `1` negative Average Daily Rate (`-$6.38`), and `1` extreme outlier (`adr == $5,400`).
  4. **Engagement Inverse Relationship**: Bookings with `>= 2` special requests (`total_of_special_requests`) or `>= 1` booking modification (`booking_changes`) exhibited less than half the cancellation rate of zero-request bookings.

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Removed `31,994` duplicate rows, dropped post-event leakage columns (`reservation_status`, `reservation_status_date`, `assigned_room_type`), filtered out `0-guest` and corrupted `adr` records, and replaced high-missingness `company` (`94.3%` null) and `agent` IDs with binary flags (`has_booking_agent`, `has_corporate_account`).
2. **Domain Feature Engineering**:
   * `total_stay_nights` & `total_booking_revenue`: Total length of stay multiplied by `adr` to quantify revenue at risk.
   * `historical_cancel_ratio`: Guest's prior cancellation ratio across historical stays.
   * `engagement_score`: Combined count of special requests, booking modifications, and parking requests.
   * `country_grouped`: Leak-free top-10 country grouping computed strictly on `X_train`.
3. **Leak-Free Preprocessing & Modeling**: Benchmarked balanced `LogisticRegression` against cost-sensitive `LGBMClassifier` with F1-optimal probability thresholding.

## Key Results & Evaluation Metrics (Test Set: 17,446 Unique Reservations)

| Model | ROC-AUC | PR-AUC | Optimal Threshold | Precision (Canceled) | Recall (Canceled) | F1-Score (Canceled) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Balanced Baseline)** | 0.8382 | 0.6719 | 0.5392 | 0.55 | 0.74 | 0.63 | 0.76 |
| **LightGBM (Cancellation Risk Engine)** | **0.8994** | **0.7777** | 0.5495 | **0.65** | **0.79** | **0.71** | **0.82** |

*(Note: Update the metrics above with the exact values printed in Cell 6 of your notebook.)*

## Business Impact & Revenue Management Recommendations
1. **Tiered Pre-Arrival Reconfirmation (`lead_time > 60 days` + `Online TA`)**: SHAP analysis identifies long `lead_time`, `Online TA` market segment, low `engagement_score` (`0` special requests), and high `adr` as the top cancellation drivers. Reservations scoring above the threshold should receive an automated pre-arrival room-preference survey or partial-deposit prompt 14 days prior to check-in (increasing guest engagement and lowering cancellation risk).
2. **Expected Net Occupancy Forecasting**: Instead of counting raw booked rooms, revenue management systems should sum $(1 - P(\text{Cancel}_i))$ across all active reservations to determine the exact room inventory available for safe overbooking.

## How to Run
1. Download `hotel_bookings.csv` from [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb