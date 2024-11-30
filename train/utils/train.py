from .process_data import load_data, process_data, split_data
from .pipeline import load_pipeline
import joblib
from .version import create_latest_model_version


def run_training(data_path: str) -> str:
    # Load the data
    data = load_data(data_path)

    # Process the data
    data = process_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data, target_col="trip_duration")

    cat_features = [
        "pickup_day",
        "pickup_month",
    ]

    num_features = [
        "pickup_hour",
        "abnormal_period",
    ]

    # Load the pipeline
    pipeline = load_pipeline(cat_features, num_features)

    # Train the pipeline
    pipeline.fit(X_train[cat_features + num_features], y_train)

    path = create_latest_model_version()

    # Evaluate the pipeline and print R2 score without scientific notation
    print(f"R2 score: {pipeline.score(X_test, y_test):.6f}")

    joblib.dump(pipeline, path)
    print(f"Model saved as {path}")
    return path
