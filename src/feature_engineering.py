import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load daily aggregated coffee sales dataset."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date/time based features for seasonality learning."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for forecasting."""
    df["lag_1"] = df["daily_revenue"].shift(1)
    df["lag_7"] = df["daily_revenue"].shift(7)
    df["lag_14"] = df["daily_revenue"].shift(14)
    df["lag_30"] = df["daily_revenue"].shift(30)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window statistics (shifted to prevent leakage)."""
    
    df["rolling_mean_7"] = df["daily_revenue"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["daily_revenue"].shift(1).rolling(14).mean()
    df["rolling_mean_30"] = df["daily_revenue"].shift(1).rolling(30).mean()

    df["rolling_std_7"] = df["daily_revenue"].shift(1).rolling(7).std()
    df["rolling_std_30"] = df["daily_revenue"].shift(1).rolling(30).std()

    df["rolling_min_7"] = df["daily_revenue"].shift(1).rolling(7).min()
    df["rolling_max_7"] = df["daily_revenue"].shift(1).rolling(7).max()

    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful business ratio features."""
    
    # Average revenue per transaction
    df["avg_revenue_per_transaction"] = df["daily_revenue"] / df["transactions"]

    # Average revenue per customer
    df["avg_revenue_per_customer"] = df["daily_revenue"] / df["unique_customers"]

    # Card usage ratio
    df["card_ratio"] = df["card_transactions"] / df["transactions"]

    # Cash usage ratio
    df["cash_ratio"] = df["cash_transactions"] / df["transactions"]

    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Create forecasting target.
    horizon=1 means predict next day revenue.
    """
    df["target"] = df["daily_revenue"].shift(-horizon)
    return df


def save_data(df: pd.DataFrame, output_path: str):
    """Save engineered dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature engineered dataset saved to: {output_path}")


def main():
    input_path = "R:\Portfolio Projects\sales forecasting using ml\main\data\processed\daily_coffee_sales.csv"
    output_path = "R:\Portfolio Projects\sales forecasting using ml\main\data\processed\daily_sales_features.csv"

    df = load_data(input_path)

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ratio_features(df)
    df = add_target(df, horizon=1)

    # Drop missing values caused by lags/rolling features
    df = df.dropna().reset_index(drop=True)

    print("\n===== FEATURE ENGINEERED DATASET SUMMARY =====")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    save_data(df, output_path)


if __name__ == "__main__":
    main()