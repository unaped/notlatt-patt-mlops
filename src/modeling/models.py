import mlflow.pyfunc
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

class LRWrapper(mlflow.pyfunc.PythonModel):
    """Custom wrapper for logistic regression to return probabilities."""
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

def train_xgboost(X_train, y_train, params, n_iter=10, cv=10):
    """Train XGBoost model with hyperparameter search."""
    model = XGBRFClassifier(random_state=42)
    model_grid = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        n_jobs=-1, 
        verbose=3, 
        n_iter=n_iter, 
        cv=cv
    )
    model_grid.fit(X_train, y_train)
    return model_grid

def train_logistic_regression(X_train, y_train, params, n_iter=10, cv=3):
    """Train logistic regression with hyperparameter search."""
    model = LogisticRegression()
    model_grid = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        verbose=3, 
        n_iter=n_iter, 
        cv=cv
    )
    model_grid.fit(X_train, y_train)
    return model_grid