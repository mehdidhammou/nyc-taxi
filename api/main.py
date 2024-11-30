import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from models import Trip

from train.utils import load_latest_model, run_training

load_dotenv(".env.local")

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/ping")
def ping():
    return {"message": "Pong!"}


@app.post("/predict")
async def predict(trip: Trip):
    model, name = load_latest_model()
    res = model.predict(trip.to_pd())
    return {
        "model": name,
        "prediction": round(res[0], 2),
        "unit": "seconds",
    }


@app.get("/train")
def train():
    dataset_path = os.getenv("DATASET_PATH")
    path = run_training(dataset_path)
    return {"message": f"Training complete. Model saved as {path}"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
