import pandas as pd
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw coffee sales dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset and convert date columns to datetime format."""
    
    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Drop invalid date rows
    df = df.dropna(subset=["date", "datetime"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure money is numeric
    df["money"] = pd.to_numeric(df["money"], errors="coerce")
    df = df.dropna(subset=["money"])

    # Remove negative or zero sales values (optional but professional)
    df = df[df["money"] > 0]

    # Standardize text columns
    df["cash_type"] = df["cash_type"].str.lower().str.strip()
    df["coffee_name"] = df["coffee_name"].str.lower().str.strip()

    return df


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction-level data into daily-level revenue dataset."""

    daily_sales = df.groupby("date").agg(
        daily_revenue=("money", "sum"),
        transactions=("money", "count"),
        unique_customers=("card", "nunique"),
        cash_transactions=("cash_type", lambda x: (x == "cash").sum()),
        card_transactions=("cash_type", lambda x: (x == "card").sum()),
    ).reset_index()

    # Sort properly by date (important for time series)
    daily_sales = daily_sales.sort_values("date").reset_index(drop=True)

    return daily_sales


def save_data(df: pd.DataFrame, output_path: str):
    """Save processed dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


def main():
    input_path = "R:\Portfolio Projects\sales forecasting using ml\main\data\coffee_sales.csv"
    output_path = "R:\Portfolio Projects\sales forecasting using ml\main\data/processed/daily_coffee_sales.csv"

    df = load_data(input_path)
    df_clean = clean_data(df)
    daily_sales = aggregate_daily_sales(df_clean)

    print("\n===== DAILY SALES DATASET SUMMARY =====")
    print(daily_sales.head())
    print("\nShape:", daily_sales.shape)
    print("\nDate range:", daily_sales["date"].min(), "to", daily_sales["date"].max())
    print("\nMissing values:\n", daily_sales.isnull().sum())

    save_data(daily_sales, output_path)


if __name__ == "__main__":
    main()
