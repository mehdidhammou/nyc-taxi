import os
import re


def create_latest_model_version() -> str:
    # Get all files in the artefacts directory
    artefacts_dir = os.getenv("ARTEFACTS_DIR")
    files = os.listdir(artefacts_dir)

    # Filter for model files matching the pattern "nyc-ridge-x.x.pkl"
    versions = [re.match(r"nyc-ridge-(\d+\.\d+)\.pkl", file) for file in files]

    # Extract the version numbers, ignore None matches
    versions = [match.group(1) for match in versions if match]

    # If no versions exist, return "0.0" (first version)
    if not versions:
        return os.path.join(artefacts_dir, "nyc-ridge-0.1.pkl")

    # Get the latest version based on numeric comparison
    latest_version = max(versions, key=lambda v: tuple(map(int, v.split("."))))

    # Increment the minor version, handle "9" case for minor version
    major, minor = map(int, latest_version.split("."))
    if minor < 9:
        new_version = f"{major}.{minor + 1}"
    else:
        new_version = f"{major + 1}.0"

    # Return the full path to the new model file
    return os.path.join(artefacts_dir, f"nyc-ridge-{new_version}.pkl")
