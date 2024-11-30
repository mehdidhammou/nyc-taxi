import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    return pd.read_csv(dataset_path)


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    unique_columns = data.columns[data.nunique() == data.shape[0]]

    if len(unique_columns) > 0:
        print(f"Removing columns: {unique_columns}")
        data = data.drop(columns=unique_columns, errors="ignore")

    print(f"Removing column: dropoff_datetime")
    data = data.drop(columns=["dropoff_datetime"], errors="ignore")

    if "pickup_datetime" in data.columns and data["pickup_datetime"].dtype == "object":
        print("Converting pickup_datetime to datetime")
        data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

    # if data doesn't have pickup_date column, add it
    if "pickup_date" not in data.columns:
        print("Adding column: pickup_date")
        data["pickup_date"] = data["pickup_datetime"].dt.date

    abnormal_threshold = 6300
    abnormal_dates = (
        data.groupby("pickup_date").size().loc[lambda x: x < abnormal_threshold]
    )
    print(f"Abnormal dates: {abnormal_dates}")

    abnormal_dates_path = os.path.join(os.getcwd(), "data", "abnormal_dates.csv")
    abnormal_dates.to_csv(abnormal_dates_path)
    print(f"Abnormal dates saved to {abnormal_dates_path}")

    print("Adding column: abnormal_period")
    data["abnormal_period"] = data["pickup_date"].isin(abnormal_dates.index).astype(int)

    print("Splitting pickup_date into pickup_month, pickup_day, pickup_hour")
    data["pickup_month"] = data["pickup_datetime"].dt.month
    data["pickup_day"] = data["pickup_datetime"].dt.day
    data["pickup_hour"] = data["pickup_datetime"].dt.hour

    return data


def split_data(
    data: pd.DataFrame,
    test_size=0.2,
    target_col="trip_duration",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_col not in data.columns:
        raise ValueError(f"Target column {target_col} not in data columns")

    X, y = data.drop(columns=[target_col]), data[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42)
