import os
import re
import joblib
from glob import glob
from sklearn.pipeline import Pipeline


def load_latest_model() -> tuple[Pipeline, str]:
    artefacts_dir = os.getenv("ARTEFACTS_DIR")
    print(f"Loading models from {artefacts_dir}")
    # Using glob to filter for model files directly
    pattern = os.path.join(artefacts_dir, "nyc-ridge-*.pkl")
    print(f"Searching for model files in {pattern}")
    model_files = glob(pattern)

    if not model_files:
        raise FileNotFoundError("No saved models found in the artefacts directory.")

    # Regex pattern compiled to extract the version number
    version_pattern = re.compile(r"nyc-ridge-(\d+)\.(\d+)\.pkl")

    # Find the model with the latest version
    latest_model = max(
        model_files,
        key=lambda file: tuple(map(int, version_pattern.search(file).groups())),
    )

    # Load the latest model
    with open(latest_model, "rb") as f:
        model = joblib.load(f)

    print(f"Loaded latest model: {latest_model}")
    return model, latest_model
