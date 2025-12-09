import time
import json
import pandas as pd
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from src.config import EXPERIMENT_NAME, MODEL_NAME, ARTIFACTS_DIR


def wait_until_ready(model_name, model_version, max_attempts=10):
    """Wait until model version is ready in the registry."""
    client = MlflowClient()
    
    for _ in range(max_attempts):
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        
        if status == ModelVersionStatus.READY:
            return True
        
        time.sleep(1)
    
    return False


def get_best_experiment_run(experiment_name):
    """Get the best run from an experiment based on f1_score."""
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    print(f"Searching experiment IDs: {experiment_ids}")
    
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    
    return experiment_best


def get_best_model_from_artifacts():
    """Get the best model based on saved model results."""
    with open("./models/model_results.json", "r") as f:
        model_results = json.load(f)
    
    results_df = pd.DataFrame({
        model: val["weighted avg"] 
        for model, val in model_results.items()
    }).T
    
    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    print(f"Best model from artifacts: {best_model}")
    
    return best_model


def get_production_model(model_name):
    """Get the current production model details."""
    client = MlflowClient()
    
    prod_model = [
        model for model in client.search_model_versions(f"name='{model_name}'")
        if dict(model)['current_stage'] == 'Production'
    ]
    
    prod_model_exists = len(prod_model) > 0
    
    if prod_model_exists:
        prod_model_version = dict(prod_model[0])['version']
        prod_model_run_id = dict(prod_model[0])['run_id']
        
        print(f'Production model name: {model_name}')
        print(f'Production model version: {prod_model_version}')
        print(f'Production model run id: {prod_model_run_id}')
        
        return True, prod_model_version, prod_model_run_id
    else:
        print('No model in production')
        return False, None, None


def compare_models(experiment_best, prod_model_exists, prod_model_run_id):
    """Compare current best model with production model."""
    train_model_score = experiment_best["metrics.f1_score"]
    model_status = {}
    run_id = None
    
    if prod_model_exists:
        _, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = details["metrics"]["f1_score"]
        
        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score
        
        if train_model_score > prod_model_score:
            print(f"New model ({train_model_score:.4f}) beats production ({prod_model_score:.4f})")
            print("Registering new model")
            run_id = experiment_best["run_id"]
        else:
            print(f"Production model ({prod_model_score:.4f}) is still better")
    else:
        print("No model in production, registering first model")
        run_id = experiment_best["run_id"]
    
    print(f"Model to register: {run_id}")
    return run_id, model_status


def register_model(run_id, model_name, artifact_path):
    """Register a model in MLflow model registry."""
    if run_id is None:
        print("No model to register")
        return None
    
    print(f'Registering model from run: {run_id}')
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    # Wait for model to be ready
    wait_until_ready(model_details.name, model_details.version)
    
    model_details_dict = dict(model_details)
    print("Model registration details:")
    print(model_details_dict)
    
    return model_details_dict


def select_and_register_best_model(experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME, artifact_path=ARTIFACTS_DIR):
    """Complete model selection and registration pipeline."""
    print("\n=== Starting Model Selection ===")
    
    # Get best experiment run
    experiment_best = get_best_experiment_run(experiment_name)
    print(f"Best experiment F1 score: {experiment_best['metrics.f1_score']:.4f}")
    
    # Get best model from artifacts (for comparison)
    get_best_model_from_artifacts()
    
    # Get production model
    prod_exists, prod_version, prod_run_id = get_production_model(model_name)
    
    # Compare models
    run_id, model_status = compare_models(experiment_best, prod_exists, prod_run_id)
    
    # Register if needed
    model_details = register_model(run_id, model_name, artifact_path)
    
    print("\n=== Model Selection Complete ===")
    return model_details


if __name__ == "__main__":
    select_and_register_best_model()