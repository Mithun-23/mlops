import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from .schemas import CaloriesPredictionRequest, CaloriesPredictionResponse

# Ensure tracking URI is set so MLflow knows where to find the registry
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"

# Global dictionary to store the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model directly from the embedded artifact folder during startup
    print("Loading extracted ML model directly from folder...")
    try:
        model_path = os.path.join(BASE_DIR, "api", "model")
        model = mlflow.pyfunc.load_model(model_path)
        ml_models["model"] = model
        print("Model successfully loaded into memory!")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(
    title="Calories Burn Prediction API",
    description="API for predicting calories burned during exercise based on body metrics.",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "api", "static")), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "api", "static", "index.html"))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in ml_models}

@app.post("/predict", response_model=CaloriesPredictionResponse)
def predict_calories(request: CaloriesPredictionRequest):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")
    
    # Convert request payload to a pandas DataFrame mapping to the exact training features
    input_data = pd.DataFrame([{
        "Age": request.age,
        "Gender": request.gender,
        "Height": request.height,
        "Weight": request.weight,
        "Duration": request.duration,
        "Heart_Rate": request.heart_rate,
        "Body_Temp": request.body_temp
    }])
    
    try:
        # The MLflow PyFunc model automatically applies the Pipeline (ColumnTransformer + Model)
        prediction = ml_models["model"].predict(input_data)
        
        # Extract the float format
        predicted_cals = float(prediction[0])
        
        return CaloriesPredictionResponse(predicted_calories=round(predicted_cals, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
