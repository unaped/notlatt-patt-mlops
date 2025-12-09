"""
Analyze dataset before preprocessing.
Saves JSON fil column stats in artifacts directory.
"""
import json
import os
import pandas as pd
import sys 

sys.path.append(os.getcwd())

def analyze_dataset(df: pd.DataFrame, artifact_dir: str) -> pd.DataFrame:
    """
    Analyzes dataset columns and generates a JSON report containing:
    - column names
    - data types
    - non-null and null counts
    - unique value counts
    - sample values

    The report is saved to artifacts directory. 
    """
    info = []
    
    for col in df.columns:
        sample = None
        if not df[col].dropna().empty:
            sample = str(df[col].dropna().iloc[0])

        info.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'non_null': int(df[col].count()),
            'null': int(df[col].isna().sum()),
            'unique': int(df[col].nunique()),
            'sample': sample
        })
    
    os.makedirs(artifact_dir, exist_ok=True)
    
    file_path = os.path.join(artifact_dir, 'dataset_analysis.json')
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return pd.DataFrame(info)

if __name__ == "__main__":
    data_path = os.path.join("..", "data", "raw_data.csv")
    artifact_path = os.path.join("..", "artifacts")

    df = pd.read_csv(data_path)
    analyze_dataset(df, artifact_path)