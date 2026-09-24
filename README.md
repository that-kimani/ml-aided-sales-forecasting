# Coffee Shop Sales Forecasting with Machine Learning

## Overview

This project builds an end-to-end machine learning pipeline to **forecast daily coffee shop revenue one day ahead**. Starting from 3,636 raw transactions, the pipeline cleans data, aggregates to daily granularity, engineers 23+ time-series features, and benchmarks classical baselines against ML models.


**Best model:** `RandomForestRegressor (n_estimators=500)` — automatically selected by lowest MAE on a time-based holdout and persisted as `models/best_model.pkl`.

| Stage | Script | What it does |
|-------|--------|--------------|
| 1. Preprocessing | `src/preprocessing.py:11` | Cleans datetimes, removes duplicates/negatives, aggregates to daily revenue |
| 2. Feature Engineering | `src/feature_engineering.py:13` | Adds time, lag, rolling, and ratio features + 1-day-ahead target |
| 3. Training | `src/train_model.py:36` | Time-split (80/20), trains Naive, MA7, Linear Regression, Random Forest |
| 4. Evaluation | `src/evaluation.py:96` | MAE/RMSE/MAPE, actual vs. predicted plots, feature importance |

---


## Business Problem

For a coffee shop, knowing tomorrow's revenue helps with:

- **Staffing** — schedule baristas based on expected demand
- **Inventory** — order milk, beans, and pastries without overstocking
- **Cash flow** — anticipate low-revenue days and plan promotions
- **Anomaly detection** — flag days where actual sales deviate sharply from forecast

A naive "tomorrow = today" baseline is not enough. This project shows that a trained model using seasonality + rolling trends can significantly reduce forecast error.

---



## Dataset

**Source:** `data/coffee_sales.csv` — anonymized point-of-sale transactions

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Transaction date (`2024-03-01` to `2025-03-22`) |
| `datetime` | datetime | Exact timestamp |
| `cash_type` | categorical | `cash` / `card` |
| `card` | string | Anonymized card ID (`ANON-...`) |
| `money` | float | Transaction amount (18.12 – 40.00) |
| `coffee_name` | categorical | 8 types: Latte, Americano, Cappuccino, etc. |

**Aggregated daily dataset** (`data/processed/daily_coffee_sales.csv`) — 381 days:

```
date,daily_revenue,transactions,unique_customers,cash_transactions,card_transactions
2024-03-01,396.30,11,9,0,11
2024-03-02,228.10,7,6,1,6
...
```

**Modeling dataset** (`data/processed/daily_sales_features.csv`) — 350 rows × 29 columns after dropping NaNs from lag/rolling windows. Date range: `2024-03-31` to `2025-03-22`.

---



## Project Pipeline

```
coffee_sales.csv (3,636 rows, transaction-level)
        │
        ▼
┌──────────────────┐
│  preprocessing.py│  clean_data() + aggregate_daily_sales()
└────────┬─────────┘
         │ daily_coffee_sales.csv (381 rows, daily)
         ▼
┌──────────────────────┐
│feature_engineering.py│  time + lag + rolling + ratio + target (horizon=1)
└────────┬─────────────┘
         │ daily_sales_features.csv (350 rows × 29 cols)
         ▼
┌──────────────────┐
│  train_model.py  │  time split 80/20 → 4 models → pick best by MAE → best_model.pkl
└────────┬─────────┘
         ▼
┌──────────────────┐
│  evaluation.py   │  MAE/RMSE/MAPE + plots + feature_importance.csv + predictions.csv
└──────────────────┘
```

All rolling features use `.shift(1)` to **prevent data leakage** — `src/feature_engineering.py:37`.

---



## Modeling

**Split strategy:** Time-based split — no shuffling (`src/train_model.py:18`). First 80% for training, last 20% (70 days) for testing. This mimics real-world forecasting.

**Models compared:**

| Model | Type | Config |
|-------|------|--------|
| Naive Baseline | Rule | `pred = lag_1` |
| Moving Average 7 | Rule | `pred = rolling_mean_7` |
| Linear Regression | ML | `sklearn.linear_model.LinearRegression` |
| **Random Forest** | **ML** | `n_estimators=500, random_state=42, n_jobs=-1` — **selected as best** |

Selection criterion: **lowest MAE** on the test set (`src/train_model.py:95`). The winning model is saved to `models/best_model.pkl` via `joblib`.

---



## Results & Evaluation

Evaluation on the **holdout 70-day test set** (`outputs/predictions.csv`):

| Metric | Value |
|--------|-------|
| **MAE** | **138.04** |
| **RMSE** | **168.93** |
| **MAPE** | **46.65%** |

> Metrics computed in `src/evaluation.py:24` — MAPE handles zero-division safely. Re-run `evaluation.py` to regenerate.


## 📁 Project Structure

```
main/
├── data/
│   ├── coffee_sales.csv               # Raw transactions (3,636 rows)
│   └── processed/
│       ├── daily_coffee_sales.csv     # Daily aggregation (381 rows)
│       └── daily_sales_features.csv   # Modeling-ready (350 rows × 29 cols)
├── src/
│   ├── preprocessing.py               # Cleaning + daily aggregation
│   ├── feature_engineering.py         # Time/lag/rolling/ratio features
│   ├── train_model.py                 # Baseline + ML training + model selection
│   └── evaluation.py                  # Metrics, plots, feature importance
├── models/
│   └── best_model.pkl                 # Saved Random Forest (joblib)
├── outputs/
│   ├── predictions.csv
│   ├── feature_importance.csv
│   ├── top 15 feature importance.png
│   └── Correlation heatmap for daily sales features data.png
├── reports/
│   ├── Act vs Pred Daily Revenue.png
│   ├── Weekly Total Predicted vs Actual revenue.png
│   ├── Monthly Total Act vs Pred.png
│   └── Predictions and Visual Interpretations.xlsx
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (tested on 3.13)
- `pip` or `conda`

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/coffee-sales-forecasting.git
cd coffee-sales-forecasting/main

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Key dependencies** (`requirements.txt`): `pandas==3.0.1`, `numpy==2.4.2`, `scikit-learn==1.8.0`, `matplotlib==3.10.8`, `scipy==1.17.0`

---

## ▶️ How to Run

Run the pipeline in order — each script is self-contained with a `main()` entry point:

```bash
# Step 1: Clean + aggregate to daily
python src/preprocessing.py

# Step 2: Engineer features + create target
python src/feature_engineering.py

# Step 3: Train baselines + ML models, save best
python src/train_model.py
# Output: Train shape: (280, 27) / Test shape: (70, 27)
#         ===== MODEL COMPARISON (Lower MAE is Better) =====
#         Best model based on MAE: Random Forest

# Step 4: Evaluate + generate plots and reports
python src/evaluation.py
# Output: ===== FINAL MODEL EVALUATION =====
#         MAE / RMSE / MAPE + plots
```

> **Note:** Scripts currently use absolute Windows paths (e.g., `R:\Portfolio Projects\...`). For portability, replace with relative paths like `data/processed/daily_sales_features.csv` or use `pathlib`.

**Quick inference example:**

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")
features = pd.read_csv("data/processed/daily_sales_features.csv")
X_latest = features.drop(columns=["date", "target"]).tail(1)
print(f"Next-day revenue prediction: {model.predict(X_latest)[0]:.2f}")
```

---

## 🖼️ Visualizations

| Visualization | Location |
|---------------|----------|
| Correlation Heatmap | `outputs/Correlation heatmap for daily sales features data.png` |
| Top 15 Feature Importance | `outputs/top 15 feature importance.png` |
| Daily Actual vs Predicted | `reports/Act vs Pred Daily Revenue.png` |
| Weekly Aggregated Forecast | `reports/Weekly Total Predicted vs Actual revenue.png` |
| Monthly Aggregated Forecast | `reports/Monthly Total Act vs Pred.png` |

Add to README for GitHub display:

```markdown
![Feature Importance](outputs/top%2015%20feature%20importance.png)
![Actual vs Predicted](reports/Act%20vs%20Pred%20Daily%20Revenue.png)
```

---

## 💡 Key Learnings

- **Time-based splitting matters** — shuffling would leak future information and inflate scores.
- **Rolling statistics > raw lags** — smoothed trends are more predictive than single-day lags.
- **Baselines are essential** — without Naive/MA7 comparison, you can't tell if ML adds value.
- **Leakage prevention** — every rolling feature is `.shift(1)` before `.rolling()`.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">If you found this useful, please ⭐ the repo!</p>
