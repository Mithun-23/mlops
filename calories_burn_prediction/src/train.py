import os
import mlflow
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from data_processor import load_data, get_preprocessor, prepare_data
from evaluate import eval_metrics

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
except Exception:
    XGB_AVAILABLE = False

# Set MLflow tracking URI directly
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

def train():
    print("Loading and preparing data...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    preprocessor = get_preprocessor()

    # Define experiments
    experiment_name = "calories_burn_prediction"
    mlflow.set_experiment(experiment_name)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=50, random_state=42)

    best_rmse = float('inf')
    best_model_name = None

    print("Starting Training Loop...")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")
            
            # Create a full prediction pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Fit the pipeline
            pipeline.fit(X_train, y_train)
            
            # Predict
            predictions = pipeline.predict(X_test)
            
            # Evaluate
            rmse, mae, r2 = eval_metrics(y_test, predictions)
            
            print(f"{model_name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
            
            # Log metrics and parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log the full pipeline model (includes preprocessing)
            mlflow.sklearn.log_model(
                sk_model=pipeline, 
                artifact_path="model",
                registered_model_name="calories_burn_model" # Overwrite/add dynamically
            )

            # Keep track of best model manually
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name

    print(f"\\nBest Model: {best_model_name} with RMSE: {best_rmse:.2f}")

if __name__ == "__main__":
    train()
