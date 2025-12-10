import datetime
from scipy.stats import uniform, randint

CURRENT_DATE = datetime.datetime.now().strftime("%Y_%B_%d")
DATA_GOLD_PATH = "./data/processed/train_data_gold.csv"
DATA_VERSION = "00000"
EXPERIMENT_NAME = CURRENT_DATE
RANDOM_STATE = 42
TEST_SIZE = 0.15

# MLflow configuration
ARTIFACT_PATH = "model"
MODEL_NAME = "lead_prediction_model"
DEFAULT_MODEL_VERSION = 1

# Model hyperparameters
XGBOOST_PARAMS = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

LR_PARAMS = {
    'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    'penalty':  ["none", "l1", "l2", "elasticnet"],
    'C' : [100, 10, 1.0, 0.1, 0.01]
}