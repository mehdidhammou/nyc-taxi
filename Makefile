DOCKER_IMAGE = "fastapi-streamlit-ml"

.PHONY: build run all clean train

# Default target to build and run the services
all: build run

# Build the docker image(s) using docker-compose
build:
	docker compose build

# Start the services defined in the docker-compose.yml file
run:
	docker compose up

# Clean up build artifacts and python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Run the training script for your model
train:
	python train/main.py
