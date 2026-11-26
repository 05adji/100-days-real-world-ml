# Day 04: Real Estate Automated Valuation Model (AVM) & Spatial Pricing Engine

## Executive Summary
Accurate property valuation is foundational for mortgage underwriting, real estate investment, and instant home-buying platforms (*iBuyers*). This project builds an EDA-first Automated Valuation Model (AVM) on `21,613` residential property transactions in King County, Washington (Greater Seattle area), combining collinearity elimination, Haversine spatial distance engineering, log-target transformation (`TransformedTargetRegressor`), **XGBoost regression**, and **SHAP explainability**.

## Business Problem
Raw housing prices exhibit severe right-skewness (`skewness > 4.0`) and strong spatial autocorrelation. Standard linear regression models trained on raw dollar amounts suffer from heteroscedasticity and disproportionate sensitivity to luxury outliers. The objectives of this project are to:
1. Conduct rigorous pre-cleaning Exploratory Data Analysis (EDA) to resolve structural collinearity and data entry corruption.
2. Minimize relative valuation error (**MAPE**) and dollar error (**MAE** & **RMSE**) while maximizing explained variance (**$R^2$**).
3. Quantify the exact valuation premium of location, construction grade, and living space using **SHAP**.

## Dataset Overview & Key EDA Findings (Before Cleaning)
* **Source**: [House Sales in King County, USA (`harlfoxem/housesalesprediction` on Kaggle)](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **Raw Dimensions**: 21,613 property sales across 21 attributes (May 2014 – May 2015).
* **Critical EDA Discoveries**:
  1. **Severe Target Skewness**: Raw sale prices range from `$75,000` to `$7,700,000` with a skewness of `4.02`. Applying `log1p(price)` normalizes the target distribution (`skewness = 0.45`), stabilizing gradient updates.
  2. **The Famous `33-Bedroom` Typo & Zero-Room Records**: Identified record `id=2402100895` listing `33` bedrooms inside a `1,620 sqft` single-story home sold for `$640,000` (a typographical error for `3` bedrooms), alongside `16` records with `0` bedrooms or `0` bathrooms.
  3. **Exact Multicollinearity**: Verified that `sqft_living` is the exact sum of `sqft_above + sqft_basement` across 100% of rows (`max diff = 0`), requiring removal of `sqft_above` to prevent redundant feature splitting.
  4. **Spatial Price Gradient**: Latitude (`lat`) and postal zone (`zipcode`) drive a steep price gradient peaking around North Seattle and Bellevue (`47.62° N`).

## Methodology & Architecture
1. **Evidence-Based Cleaning**: Corrected the `33-bedroom` entry to `3`, removed `16` zero-bedroom/bathroom records, dropped `sqft_above` to eliminate exact collinearity, and cast the `70` unique `zipcode` values to categorical strings.
2. **Spatial & Domain Feature Engineering**:
   * `dist_to_seattle_center_km`: Haversine great-circle distance (in km) from each property's `(lat, long)` coordinates to the Seattle/Bellevue commercial core.
   * `living_to_neighbor15_ratio`: Ratio of the home's interior area (`sqft_living`) to the average of its 15 nearest neighbors (`sqft_living15`).
   * `house_age` & `years_since_update`: Captures physical depreciation and renovation recency at the exact date of sale.
3. **Log-Target Regression Pipeline**: Wrapped both `Ridge` and `XGBRegressor` inside `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)` with leak-free `ColumnTransformer` scaling and one-hot encoding.

## Key Results & Evaluation Metrics (Test Set: 4,320 Properties)

| Model | RMSE (USD) | MAE (USD) | MAPE (%) | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Log-Target Ridge Regression (Baseline)** | $128,241.20 | $71,987.01 | 13.45% | 0.8736 |
| **Log-Target XGBoost Regressor (AVM Engine)** | **$122,781.43** | **$60,040.67** | **11.26%** | **0.8841** |

## Business Impact & Valuation Insights
1. **Spatial Coordinates Outperform Raw Zipcodes**: SHAP analysis demonstrates that `lat`, `dist_to_seattle_center_km`, `sqft_living`, and `grade` account for the vast majority of valuation variance—confirming that continuous geospatial coordinates capture micro-neighborhood transitions that discrete zipcodes miss.
2. **Automated Underwriting Triage**: With a **~11–12% MAPE**, properties whose asking price deviates by less than 10% from the XGBoost AVM estimate can be fast-tracked for automated appraisal waivers, reserving costly in-person physical appraisals for high-variance luxury estates.

## How to Run
1. Download `kc_house_data.csv` from [Kaggle House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) and place it in `data/raw/`.
2. Install dependencies and execute the notebook:
   ```bash
   pip install -r requirements.txt
   jupyter notebook src/notebook.ipynb