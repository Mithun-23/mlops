# Calories Burn Prediction System

This is a complete end-to-end Machine Learning web application that predicts calories burned during a workout session using Scikit-learn, MLflow, FastAPI, and Docker.

## Project Structure
- **`data/`**: Processed and generated datasets.
- **`src/`**: Machine Learning logic. `data_processor.py`, `train.py`, `evaluate.py`.
- **`api/`**: FastAPI implementation (`main.py`, `schemas.py`) to expose the trained model.
- **`mlruns/` & `mlflow.db`**: Local SQLite model registry populated by the MLflow tracker.
- **`Dockerfile`**: Container definition to deploy the backend seamlessly.

## 1. Local Development Setup

To test from scratch:

```bash
# Set up a virtual env and install requirements
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate the synthetic data identical to the target set
python3 src/generate_data.py

# Train models (Random Forest, Linear Regression) & log to MLflow tracking
PYTHONPATH=src python3 src/train.py

# Test the FastAPI endpoint locally 
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## 2. Testing the API
Open another terminal:
```bash
curl -X POST "http://0.0.0.0:8000/predict" \\
-H "Content-Type: application/json" \\
-d '{"age": 28, "gender": "female", "height": 165.0, "weight": 60.5, "duration": 45, "heart_rate": 145, "body_temp": 39.2}'
```
*Expected Response:* `{"predicted_calories": 262.88}` (Actual value depends on latest model registry state).

## 3. Web Deployment (Railway)

This codebase is natively containerized, making it incredibly simple to deploy in the cloud using [Railway.app](https://railway.app/).

1. **Push to GitHub**: Push this entire project folder to a public or private GitHub repository.
2. **Link Railway**: Create an account on Railway and click "New Project" -> "Deploy from GitHub Repo".
3. **Select Repository**: Pick the repo you just pushed.
4. **Auto-Build**: Railway will automatically detect the `Dockerfile` and build the container image.
5. **Get Public URL**: Once deployed, Railway maps the `PORT` env var dynamically and exposes your API on a public hyperlink (e.g. `https://calories-burn-app.up.railway.app`). You can immediately POST to `https://<YOUR-URL>/predict`.

*Note: Since the `.gitignore` currently ignores `mlruns/` and `mlflow.db`, you should EITHER remove those from `.gitignore` so the pre-trained model pushes to GitHub and builds with your container, OR configure your Railway container to pull dynamically from a cloud-hosted MLflow bucket upon startup.*
