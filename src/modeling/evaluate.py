import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    classification_report
)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Evaluate model performance on train and test sets."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    results = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy train: {results['train_accuracy']:.4f}")
    print(f"Accuracy test: {results['test_accuracy']:.4f}")
    
    return results

def print_confusion_matrix(y_true, y_pred, dataset_name="Test"):
    """Print confusion matrix and classification report."""
    print(f"\n{dataset_name} actual/predicted")
    print(pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))
    print("\nClassification report")
    print(classification_report(y_true, y_pred))