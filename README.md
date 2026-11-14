# 100 Days of Real-World Machine Learning Projects (1 Day 1 Project)

Welcome to the **100 Days of Real-World Machine Learning Projects** portfolio repository! This repository documents an intensive 100-day engineering challenge—building and shipping **one end-to-end real-world Machine Learning project every single day** (November 13, 2026 – February 20, 2027), spanning business tabular modeling, recommender systems, time series forecasting, NLP, computer vision, and production MLOps.

## Program Objectives

1. **Solve Real Business Problems**: Build data-driven solutions evaluated against industry-aligned metrics (ROC-AUC, PR-AUC, RMSE, F1-Score, and expected financial impact).
2. **Master Full-Spectrum Machine Learning**: Gain hands-on depth across Supervised & Unsupervised Learning, Time Series Forecasting, Natural Language Processing (NLP), Computer Vision, and End-to-End MLOps.
3. **Production-Grade Code & Documentation**: Maintain clean, modular, reproducible code and comprehensive technical documentation for every project.

## Repository Structure

```text
100-days-real-world-ml/
├── README.md
├── .gitignore
├── requirements.txt
├── day-01-telco-customer-churn/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── src/
│   │   └── train.py
│   └── requirements.txt
├── day-02-credit-risk-scoring/
│   └── ...
└── ...
```

## 100-Day Curriculum Roadmap

| Phase | Days | Core Domain | Key Real-World Applications |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Days 1–20 | Supervised Learning & Business Tabular Data | Customer Churn Prediction, Credit Risk Scoring, Fraud Detection, Dynamic Pricing, Lead Conversion, Insurance Claim Prediction |
| **Phase 2** | Days 21–40 | Unsupervised Learning & Recommender Systems | RFM Customer Segmentation, Predictive Maintenance Anomaly Detection, Market Basket Analysis, Collaborative & Content-Based Filtering |
| **Phase 3** | Days 41–60 | Time Series Forecasting & Sequential Data | Multi-Store Demand Forecasting, Power Grid Load Prediction, Server Traffic Anomaly Detection, Supply Chain Inventory Optimization |
| **Phase 4** | Days 61–75 | Natural Language Processing (NLP) | E-Commerce Sentiment Analysis, Automated Support Ticket Triage, Financial Contract NER, Document Summarization, Semantic Search & RAG |
| **Phase 5** | Days 76–90 | Computer Vision | Manufacturing Surface Defect Detection, Plant Disease Classification, Automated Receipt OCR, Vehicle Counting, Construction PPE Safety Detection |
| **Phase 6** | Days 91–100 | End-to-End MLOps & Capstone Deployment | FastAPI Model Serving, Docker Containerization, Interactive Streamlit/Gradio Dashboards, Data Drift Monitoring, Automated Retraining Pipelines |

## Daily Project Log (Progress Tracker)

### Phase 1: Supervised Learning & Business Tabular Data (Days 1–20)

| Day | Date | Project Title | Business Use Case & Domain | Core Algorithms / Models | Key Metrics | Project Directory | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | Nov 13, 2026 | Telco Customer Churn Prediction | Telecom Customer Retention | Logistic Regression, XGBoost, SHAP | ROC-AUC: 0.8419, PR-AUC: 0.6615 | [`./day-01-telco-customer-churn`](./day-01-telco-customer-churn) | ✅ Completed |
| 02 | Nov 14, 2026 | Credit Risk & Loan Default Scoring | Consumer Lending & Banking Risk | Logistic Regression, LightGBM, SHAP | ROC-AUC: 0.9533, PR-AUC: 0.9131, KS: 0.7747 | [`./day-02-credit-risk-scoring`](./day-02-credit-risk-scoring) | ✅ Completed |
| 03 | Nov 15, 2026 | Credit Card Fraud Detection | Financial Transaction Security | Isolation Forest, CatBoost, SMOTE | PR-AUC, F1-Score | `./day-03-fraud-detection` | ⏳ Planned |
| 04 | Nov 16, 2026 | Real Estate Valuation Engine | Residential Property Appraisal | Ridge, Random Forest, XGBoost | RMSE, MAE, R² | `./day-04-house-price-valuation` | ⏳ Planned |
| 05 | Nov 17, 2026 | E-Commerce Delivery Time Estimation | Logistics & Last-Mile Optimization | Gradient Boosting, Feature Hashing | MAE, MAPE | `./day-05-delivery-time-estimation` | ⏳ Planned |
| 06–20 | Nov 18 – Dec 02, 2026 | *Updated daily at 07:00 AM (UTC+7)* | Business Classification & Regression | Ensemble & Interpretable ML | Case-Specific | - | ⏳ Planned |

### Phase 2: Unsupervised Learning & Recommender Systems (Days 21–40)

| Day | Date | Project Title | Business Use Case & Domain | Core Algorithms / Models | Key Metrics | Project Directory | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 21 | Dec 03, 2026 | E-Commerce Customer Segmentation | Personalized Lifecycle Marketing (RFM) | K-Means, Hierarchical Clustering, UMAP | Silhouette Score | `./day-21-customer-segmentation` | ⏳ Planned |
| 22 | Dec 04, 2026 | Industrial Sensor Anomaly Detection | Manufacturing Predictive Maintenance | Isolation Forest, Autoencoder | Precision@K, F1 | `./day-22-sensor-anomaly-detection` | ⏳ Planned |
| 23 | Dec 05, 2026 | Retail Market Basket Analysis | Cross-Selling & Product Bundling | Apriori, FP-Growth | Lift, Confidence | `./day-23-market-basket-analysis` | ⏳ Planned |
| 24–40 | Dec 06–22, 2026 | *Updated daily at 07:00 AM (UTC+7)* | Clustering & Personalization Engines | Matrix Factorization, Two-Tower, Graph | MAP@K, NDCG@K | - | ⏳ Planned |

### Phase 3: Time Series Forecasting & Sequential Data (Days 41–60)

| Day | Date | Project Title | Business Use Case & Domain | Core Algorithms / Models | Key Metrics | Project Directory | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 41 | Dec 23, 2026 | Multi-Store Retail Demand Forecasting | Inventory & Supply Planning | SARIMAX, Prophet, LightGBM Lag | WAPE, RMSE | `./day-41-retail-demand-forecasting` | ⏳ Planned |
| 42 | Dec 24, 2026 | Hourly Power Grid Load Forecasting | Energy Distribution Management | XGBoost Time Series, LSTM | MAPE, MAE | `./day-42-energy-load-forecasting` | ⏳ Planned |
| 43–60 | Dec 25, 2026 – Jan 11, 2027 | *Updated daily at 07:00 AM (UTC+7)* | Sequential Modeling & Forecasting | Prophet, N-BEATS, Temporal Fusion | MASE, sMAPE | - | ⏳ Planned |

### Phase 4: Natural Language Processing (NLP) (Days 61–75)

| Day | Date | Project Title | Business Use Case & Domain | Core Algorithms / Models | Key Metrics | Project Directory | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 61 | Jan 12, 2027 | Marketplace Product Review Sentiment | Voice-of-Customer Analytics | TF-IDF, Fine-tuned RoBERTa/IndoBERT | Macro F1-Score | `./day-61-review-sentiment-analysis` | ⏳ Planned |
| 62 | Jan 13, 2027 | Automated Customer Ticket Triage | Customer Support Routing Automation | Transformer Text Classification | Accuracy, F1 | `./day-62-support-ticket-triage` | ⏳ Planned |
| 63–75 | Jan 14–26, 2027 | *Updated daily at 07:00 AM (UTC+7)* | Information Extraction & RAG Systems | BERT NER, Dense Embeddings, Vector DB | Recall@K, ROUGE | - | ⏳ Planned |

### Phase 5: Computer Vision (Days 76–90)

| Day | Date | Project Title | Business Use Case & Domain | Core Algorithms / Models | Key Metrics | Project Directory | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 76 | Jan 27, 2027 | Manufacturing Surface Defect Detection | Automated Visual Quality Control | ResNet50, EfficientNet Transfer Learning | mAP, F1-Score | `./day-76-surface-defect-detection` | ⏳ Planned |
| 77 | Jan 28, 2027 | Construction Site PPE Safety Detection | Workplace Safety Compliance | YOLOv8 Object Detection | mAP@0.5 | `./day-77-ppe-safety-detection` | ⏳ Planned |
| 78–90 | Jan 29 – Feb 10, 2027 | *Updated daily at 07:00 AM (UTC+7)* | Classification, Object Detection & OCR | CNN, Vision Transformer, PaddleOCR | IoU, CER/WER | - | ⏳ Planned |

### Phase 6: End-to-End MLOps & Capstone Deployment (Days 91–100)

| Day | Date | Project Title | Business Use Case & Domain | Core Algorithms / Models | Key Metrics | Project Directory | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 91 | Feb 11, 2027 | Production Model Serving API | Real-Time Inference Microservice | FastAPI, Pydantic, Uvicorn | Latency (ms), RPS | `./day-91-fastapi-model-serving` | ⏳ Planned |
| 92–100 | Feb 12–20, 2027 | *Updated daily at 07:00 AM (UTC+7)* | Docker, CI/CD, Drift Monitoring & Capstone | Docker, EvidentlyAI, MLflow, Streamlit | End-to-End SLA | - | ⏳ Planned |

## Getting Started Locally

1. **Clone this repository**:
   ```bash
   git clone https://github.com/05adji/100-days-real-world-ml.git
   cd 100-days-real-world-ml
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or: venv\Scripts\activate  # On Windows
   ```

3. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Navigate to any daily project directory**:
   ```bash
   cd day-01-telco-customer-churn
   jupyter notebook
   ```

## Recommended `.gitignore` Configuration

Save the following rules into a `.gitignore` file in your repository root to prevent large raw datasets, virtual environments, and model binaries from bloating your Git history:

```gitignore
# Virtual Environment & Python Cache
venv/
.venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# Large Datasets & Model Artifacts
*.csv
*.parquet
*.zip
*.pkl
*.joblib
*.h5
*.pt
*.onnx
data/raw/
data/processed/

# Environment Variables & OS Files
.env
.DS_Store
Thumbs.db
```

## Author & Contact

* **Name**: `Putra Adji Syachguna`
* **LinkedIn**: `www.linkedin.com/in/putra-adji-syachguna`
* **GitHub**: `https://github.com/05adji`
