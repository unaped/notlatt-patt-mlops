"""
Data ingestion: loads raw data from CSV.
"""

import pandas as pd
import os
import sys

sys.path.append(os.getcwd())

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads the raw dataset from a CSV file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Can't find input file: {path}")
    df = pd.read_csv(path)
    print(df.head())
    return df


if __name__ == "__main__":
    full_path = os.path.join("..", "data", "raw_data.csv")
    df = load_raw_data(full_path)
