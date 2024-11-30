import os

from dotenv import load_dotenv

from train.utils.train import run_training

if __name__ == "__main__":
    load_dotenv(".env.local")
    dataset_path = os.getenv("DATASET_PATH")
    run_training(dataset_path)
