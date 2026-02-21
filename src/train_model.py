import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split_time(df: pd.DataFrame, split_ratio: float = 0.8):
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n===== {model_name} Performance =====")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    return mae, rmse


def main():
    input_path = r"R:\Portfolio Projects\sales forecasting using ml\main\data\processed\daily_sales_features.csv"
    model_output_path = r"R:\Portfolio Projects\sales forecasting using ml\main\models\best_model.pkl"

    df = load_data(input_path)

    # Train/Test Split
    train, test = train_test_split_time(df, split_ratio=0.8)

    # Features and target
    X_train = train.drop(columns=["date", "target"])
    y_train = train["target"]

    X_test = test.drop(columns=["date", "target"])
    y_test = test["target"]

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ==========================
    # BASELINE MODELS
    # ==========================

    # Baseline 1: Naive (tomorrow = today)
    naive_preds = test["lag_1"]
    naive_mae, naive_rmse = evaluate_model(y_test, naive_preds, "Naive Baseline (lag_1)")

    # Baseline 2: Rolling Mean 7
    ma7_preds = test["rolling_mean_7"]
    ma7_mae, ma7_rmse = evaluate_model(y_test, ma7_preds, "Moving Average Baseline (7-day)")

    # ==========================
    # ML MODELS
    # ==========================

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_mae, lr_rmse = evaluate_model(y_test, lr_preds, "Linear Regression")

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mae, rf_rmse = evaluate_model(y_test, rf_preds, "Random Forest Regressor")

    # ==========================
    # SELECT BEST MODEL
    # ==========================

    results = {
        "Naive Baseline": naive_mae,
        "Moving Avg 7": ma7_mae,
        "Linear Regression": lr_mae,
        "Random Forest": rf_mae
    }

    best_model_name = min(results, key=results.get)
    print("\n===== MODEL COMPARISON (Lower MAE is Better) =====")
    for name, mae in results.items():
        print(f"{name}: {mae:.3f}")

    print(f"\nBest model based on MAE: {best_model_name}")

    # Save best model
    os.makedirs("models", exist_ok=True)

    if best_model_name == "Linear Regression":
        joblib.dump(lr_model, model_output_path)
    elif best_model_name == "Random Forest":
        joblib.dump(rf_model, model_output_path)
    else:
        # Baselines are not saved because they are rule-based
        print("Best model is a baseline. No model saved.")
        return

    print(f"\nBest model saved to: {model_output_path}")


if __name__ == "__main__":
    main()
