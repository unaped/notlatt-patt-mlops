import pandas as pd
import os
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler  

sys.path.append(os.getcwd())
from .utils import load_params

def bin_source(df: pd.DataFrame, binning_map: dict) -> pd.DataFrame:
    df['bin_source'] = df['source'].map(binning_map)
    df['bin_source'] = df['bin_source'].fillna('other')
    return df

def standardize_data(df: pd.DataFrame, num_cols: list, artifact_dir: str, scaler_name: str) -> pd.DataFrame:
    valid_cols = [c for c in num_cols if c in df.columns]
    
    scaler = MinMaxScaler()
    df[valid_cols] = scaler.fit_transform(df[valid_cols])
    
    scaler_path = os.path.join(artifact_dir, scaler_name)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return df

def run_features(params: dict) -> pd.DataFrame:
    dirs = params['directories']
    fe = params['feature_engineering']
    schema = params['schema']
    
    input_path = os.path.join(".", "data", "silver_data.csv")
    df = pd.read_csv(input_path)
    
    # Bin
    df = bin_source(df, fe['source_binning_map'])
    
    # Drop
    df = df.drop(columns=fe['drop_features'], errors='ignore')
    
    # Scale
    df = standardize_data(
        df, 
        num_cols=schema['continuous_features'], 
        artifact_dir=dirs['artifact_dir'], 
        scaler_name=fe['scaler_path']
    )
    
    output_path = os.path.join(".", "data", "gold_data.csv")    
    os.makedirs(dirs['artifact_dir'], exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Successfully created gold data.")
    return df

if __name__ == "__main__":
    params = load_params("params.yaml")
    df_gold = run_features(params)