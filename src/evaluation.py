import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

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


def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

    return mae, rmse, mape


def plot_actual_vs_predicted(dates, y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual Revenue")
    plt.plot(dates, y_pred, label="Predicted Revenue")
    plt.title("Actual vs Predicted Daily Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred

    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residual Plot (Actual - Predicted)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Residual Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("\nThis model does not support feature importance.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return feature_importance_df


def save_predictions(test_df, y_pred, output_path= r"R:\Portfolio Projects\sales forecasting using ml\main\reports\predictions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = test_df[["date", "daily_revenue", "target"]].copy()
    results["predicted_target"] = y_pred
    results["error"] = results["target"] - results["predicted_target"]

    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")


def main():
    data_path = r"R:\Portfolio Projects\sales forecasting using ml\main\data\processed\daily_sales_features.csv"
    model_path = r"R:\Portfolio Projects\sales forecasting using ml\main\models\best_model.pkl"

    df = load_data(data_path)
    train, test = train_test_split_time(df, split_ratio=0.8)

    X_test = test.drop(columns=["date", "target"])
    y_test = test["target"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    mae, rmse, mape = evaluate_metrics(y_test, y_pred)

    print("\n===== FINAL MODEL EVALUATION =====")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%")

    # Plots
    plot_actual_vs_predicted(test["date"], y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # Feature importance
    feature_names = X_test.columns
    feature_importance_df = plot_feature_importance(model, feature_names, top_n=15)

    if feature_importance_df is not None:
        os.makedirs("reports", exist_ok=True)
        feature_importance_df.to_csv(r"R:\Portfolio Projects\sales forecasting using ml\main\reports\feature_importance.csv", index=False)
        print("\nFeature importance saved to: reports/feature_importance.csv")

    # Save predictions
    save_predictions(test, y_pred, output_path=r"R:\Portfolio Projects\sales forecasting using ml\main\reports\predictions.csv")


if __name__ == "__main__":
    main()
