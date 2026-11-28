# Day 16: Used Car Dynamic Valuation & Depreciation Engine

## Executive Summary
Used vehicle marketplaces require fast, transparent resale price estimates across diverse makes, model years, and engine specifications. Raw automotive listings frequently contain massive dealer-listing duplication and dirty unit-suffixed specification strings (`kmpl`, `CC`, `bhp`, `Nm@rpm`). This project builds an EDA-first valuation pipeline on `8,128` CarDekho listings (`6,926` unique vehicles after deduplication), combining regex telemetry parsing, log-target regression (`TransformedTargetRegressor`), **LightGBM**, and **SHAP explainability**.

## Business Problem
In automotive recommerce, pricing errors directly impact gross margins (overpaying on vehicle acquisition) and inventory turnover (overpricing on retail listings). The objectives of this project are to:
1. Audit raw dealer and private-seller listings prior to cleaning to remove duplicated inventory that leaks across train/test splits.
2. Parse heterogeneous text specifications (`mileage`, `engine`, `max_power`, `torque`) safely into clean numerical features.
3. Minimize relative valuation error (**MAPE**) and dollar/rupee error (**MAE**, **RMSE**) across economy and luxury tiers.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [Vehicle Dataset from CarDekho (`nehalbirla/vehicle-dataset-from-cardekho` on Kaggle)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (`Car details v3.csv`)
* **Raw Dimensions**: 8,128 vehicle listings across 13 attributes.
* **Critical EDA Discoveries**:
  1. **Severe Dealer Duplicate Distortion (`1,202` Rows / 14.8%)**: Exact-row auditing revealed `1,202` duplicate listings concentrated heavily among commercial dealers—including `209` out of `236` `Trustmark Dealer` rows. Removing duplicates lowers the dataset mean price from `₹638k` to `₹517k` and eliminates train-test leakage.
  2. **Dirty Unit-Suffixed Strings & Empty Unit Rows**: `mileage`, `engine`, `max_power`, and `torque` are stored as strings with mixed units (`kmpl`, `km/kg`, `Nm`, `kgm`), alongside malformed `" bhp"` strings missing numeric values.
  3. **Right-Skewed Valuation Tail**: Resale prices span `₹29,999` to `₹10,000,000` (`skewness > 4.0`), normalized cleanly via `log1p` target transformation.

## Methodology & Architecture
1. **Evidence-Based Cleaning & Regex Extraction**: Removed `1,202` exact duplicates, extracted clean floats from `mileage`, `engine`, and `max_power`, standardized `torque` into Newton-meters (`kgm × 9.80665`), and extracted vehicle `brand` from `name`.
2. **Automotive Feature Engineering**:
   * `vehicle_age`, `km_per_year`, & `log_km_driven`: Captures age-driven depreciation and annual usage intensity.
   * `power_per_liter`: Specific engine output (`max_power_bhp / engine_L`) separating naturally aspirated economy cars from turbocharged/luxury models.
3. **Leak-Free Log-Target Benchmarking**: Benchmarked `Log-Target Ridge` against `Log-Target LightGBM` using median imputation, `OrdinalEncoder` for ownership tier, and rare-brand grouping (`< 1%`) computed strictly inside `X_train`.

## Key Results & Evaluation Metrics (Test Set: 1,386 Unique Vehicles)

| Model | MAE (INR) | RMSE (INR) | MAPE (%) | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Log-Target Ridge Regression (Baseline)** | ₹105,107.23 | ₹227,665.92 | 21.60% | 0.8223 |
| **Log-Target LightGBM (Used Car AVM)** | **₹75,377.16** | **₹147,257.29** | **16.70%** | **0.9257** |

### Error Analysis & Model Limitations (Residual Audit)
Inspecting the top 5 largest valuation errors on the test set exposed three concrete root causes behind residual outliers:
1. **Median Imputation Distortion on Missing Power (`max_power_bhp = NaN`)**: Two of the top three absolute errors (`₹2.15M` vehicles) had missing `max_power_bhp` values; global median imputation (`~82 bhp`) caused the model to severely undervalue them (`₹290k` and `₹636k`). *Future fix: Group-wise imputation by `brand` and `engine_cc`.*
2. **Rare-Brand Grouping & Dropped Model Name**: Three of the top five absolute errors fell under `Other_Brand` (`< 1%` frequency), while a `2014 Toyota (170.3 bhp)` priced at `₹3.50M` was predicted at `₹1.53M` because only the top-level `brand` was extracted rather than the specific vehicle model (`Fortuner`/`Land Cruiser`).
3. **Unobserved Physical Condition**: In percentage-error audits, a `2020 Volkswagen Diesel` listed at `₹260k` (predicted `₹680k`) and older `2006–2008` vehicles reflect unrecorded accident history, mechanical condition, or seller pricing typos absent from specification tables.

## Business Impact & Marketplace Insights
1. **Depreciation & Powertrain Drivers**: SHAP confirms `year` (`vehicle_age`), `max_power_bhp`, `km_driven`, and `brand` as the dominant resale valuation drivers—allowing instant automated price guardrails for consumer C2B trade-in quotes.
2. **Clean Inventory Indexing**: Deduplicating dealer bulk uploads prior to model training prevents luxury outliers from distorting automated pricing across mainstream segments.

## How to Run
1. Download `Car details v3.csv` from [Kaggle Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb