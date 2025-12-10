"""
Helper functions. Don't know if I need all of them tho.
"""

import yaml
import json
import os
import pandas as pd

def load_params(path: str = "params.yaml") -> dict:
    """
    Reads the YAML file and returns a standard Python dictionary.
    """
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return {}
    
def save_json(data: dict, path: str):
    """
    Save dictionary as JSON to a specific path.
    Creates the directory if it doesn't exist.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved metadata to: {path}")

def describe_numeric_col(df: pd.Series) -> pd.Series:
    """
    Generates descriptive stats for a numeric series.
    Used for outlier summaries.
    """
    return pd.Series(
        [df.count(), df.isnull().sum(), df.mean(), df.min(), df.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )