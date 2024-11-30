# NYC Ridge Prediction Model

This project is a machine learning application designed to predict trip durations in New York City using a Ridge regression model. The application includes a FastAPI backend for model training and prediction, and a Streamlit frontend for user interaction.

## Setup

### Prerequisites

- Docker
- Docker Compose
- Python 3.13

### Installation

1. Configure`.env.local` file variables.

2. Build and run the Docker containers:
   ```sh
   make all
   ```

## Usage

### API Endpoints

- **Root**: `GET /`
  - Returns a welcome message.
- **Ping**: `GET /ping`
  - Returns a pong message to check API health.
- **Predict**: `POST /predict`
  - Accepts a JSON payload with trip details and returns the predicted trip duration.
- **Train**: `GET /train`
  - Triggers the model training process and saves the trained model.

### Streamlit Frontend

1. Access the Streamlit frontend at `http://localhost:8501`.
2. Use the sliders to input trip details and click "Predict" to get the trip duration prediction.
3. Click "Train Model" to trigger the model training process.

## License

This project is licensed under the MIT License.
