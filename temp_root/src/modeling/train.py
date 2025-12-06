import os
import json
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pprint import pprint

from src.config import *
from src.features import prepare_training_data
from src.modeling.models import train_xgboost, train_logistic_regression, LRWrapper
from src.modeling.evaluate import evaluate_model, print_confusion_matrix

def setup_directories():
    """Create necessary directories."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)

def split_data(data, target_col='lead_indicator'):
    """Split data into features and target."""
    y = data[target_col]
    X = data.drop([target_col], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=TEST_SIZE, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def save_artifacts(X_train, model_results):
    """Save column list and model results."""
    # Save column list
    column_list_path = './models/columns_list.json'
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        pprint(columns)  # Print what columns are being saved
        json.dump(columns, columns_file)
    print(f'Saved column list to {column_list_path}')
    
    # Save model results
    model_results_path = "./models/model_results.json"
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)
    print(f'Saved model results to {model_results_path}')

def train_all_models():
    """Main training pipeline."""
    setup_directories()
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Prepare data
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    data = prepare_training_data(DATA_GOLD_PATH, cat_cols)
    X_train, X_test, y_train, y_test = split_data(data)
    
    model_results = {}
    
    # Train XGBoost
    print("\n=== Training XGBoost ===")
    xgboost_grid = train_xgboost(X_train, y_train, XGBOOST_PARAMS)
    xgboost_model = xgboost_grid.best_estimator_
    xgboost_model_path = "./models/lead_model_xgboost.json"
    xgboost_model.save_model(xgboost_model_path)
    
    xgb_results = evaluate_model(xgboost_grid, X_train, X_test, y_train, y_test, "XGBoost")
    model_results[xgboost_model_path] = xgb_results['classification_report']
    
    print_confusion_matrix(y_test, xgboost_grid.predict(X_test), "XGBoost Test")
    print_confusion_matrix(y_train, xgboost_grid.predict(X_train), "XGBoost Train")

    print("Best XGBoost params:")
    pprint(xgboost_grid.best_params_)
    
    # Train Logistic Regression with MLflow
    print("\n=== Training Logistic Regression ===")
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        lr_grid = train_logistic_regression(X_train, y_train, LR_PARAMS)
        lr_model = lr_grid.best_estimator_
        lr_model_path = "./models/lead_model_lr.pkl"
        
        lr_results = evaluate_model(lr_grid, X_train, X_test, y_train, y_test, "Logistic Regression")
        model_results[lr_model_path] = lr_results['classification_report']
        
        # Log to MLflow
        mlflow.log_metric('f1_score', f1_score(y_test, lr_grid.predict(X_test)))
        mlflow.log_param("data_version", DATA_VERSION)
        
        # Save model
        joblib.dump(value=lr_model, filename=lr_model_path)
        mlflow.pyfunc.log_model('model', python_model=LRWrapper(lr_model))
        
        print_confusion_matrix(y_test, lr_grid.predict(X_test), "LR Test")
        print_confusion_matrix(y_train, lr_grid.predict(X_train), "LR Train")

        print("Best Logistic Regression params:")
        pprint(lr_grid.best_params_)
    
    # Save artifacts
    save_artifacts(X_train, model_results)
    
    print("\n=== Training Complete ===")
    return model_results

if __name__ == "__main__":
    train_all_models()