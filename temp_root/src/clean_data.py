"""
Data cleaning:
- Filters by date and source.
- Drops columns that are not used in modeling.
- Removes rows with empty target values.
- Handles outliers and imputes missing values.
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from src.utils import load_params, save_json, describe_numeric_col
from src.analyze_data import analyze_dataset # Am I double-using this lol

# OUTLIERS
def handle_outliers(df: pd.DataFrame, 
                    numeric_cols: list, 
                    threshold: float, 
                    artifact_dir: str) -> pd.DataFrame:
    """
    Clips outliers based on standard deviation and saves outlier summary.
    
    Args:
        df: DataFrame to process
        numeric_cols: List of numeric column names to check for outliers
        threshold: Standard deviation threshold
        artifact_dir: Directory path for JSON outlier summary
    
    Returns:
        DataFrame with outliers clipped
    """
    outlier_summary = {}
    
    for col in numeric_cols:
        if col in df.columns:
            # Define data limits
            mean = df[col].mean()
            std = df[col].std()
            upper_limit = mean + (threshold * std)
            lower_limit = mean - (threshold * std)
            
            # Save data limits before outlier removal
            original_min = float(df[col].min())
            original_max = float(df[col].max())
            
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
            
            # Saves data limits after outlier removal
            outlier_summary[col] = {
                "original_min": original_min,
                "original_max": original_max,
                "new_min": float(df[col].min()),
                "new_max": float(df[col].max()),
                "upper_limit": float(upper_limit),
                "lower_limit": float(lower_limit)
            }
            
    # Save JSON file
    save_json(outlier_summary, os.path.join(artifact_dir, "outlier_summary.json"))
    
    return df

# IMPUTE MISSING VALUES
def impute_missing_values(df: pd.DataFrame, 
                          numeric_cols: list, 
                          cat_cols: list, 
                          strategy: str, 
                          custom_values: dict) -> pd.DataFrame:
    """
    Fills missing values based on column type and strategy.
    
    Args:
        df: DataFrame to process
        numeric_cols: List of continuous column names
        cat_cols: List of categorical column names
        strategy: Imputation strategy for continuous columns ('mean' or 'median')
        custom_values: Dictionary mapping column names to specific fill values
    
    Returns:
        DataFrame with missing values imputed
    """
    
    for col, value in custom_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Categorical: fill with Mode
    for col in cat_cols:
        if col in df.columns and not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Numeric: fill with Mean/Median
    for col in numeric_cols:
        if col in df.columns:
            if strategy == "mean":
                fill_val = df[col].mean()
            elif strategy == "median":
                fill_val = df[col].median()
            else:
                fill_val = 0 
            df[col] = df[col].fillna(fill_val)
            
    return df

# FILTER DATA BY DATE RANGE
def filter_by_date(df: pd.DataFrame, 
                   min_date: str, 
                   max_date: str, 
                   date_col: str,
                   artifact_dir: str) -> pd.DataFrame:
    """
    Filters DataFrame by date range and saves date limits.
    
    Args:
        df: DataFrame to filter
        min_date: Minimum date string (YYYY-MM-DD format)
        max_date: Maximum date string (YYYY-MM-DD format)
        date_col: Name of the date column
        artifact_dir: Directory pat for JSON actual date limits
    
    Returns:
        Filtered DataFrame
    """
    # Change data type of input date strings limits and date column
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    min_date = pd.to_datetime(min_date).date()
    max_date = pd.to_datetime(max_date).date() 

    # Apply the date range
    df_filtered= df[(df[date_col] >= min_date) & (df[date_col] <= max_date)]

    # Save actual date limits
    date_limits = {
        "min_date": str(df_filtered[date_col].min()), 
        "max_date": str(df_filtered[date_col].max())
    }

    save_json(date_limits, os.path.join(artifact_dir, "date_limits.json"))
    
    return df_filtered

# COMBINE CLEANING STEPS
# Is it a good idea to use YAML here? 
def clean_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Executes the complete data cleaning pipeline.
    
    Args:
        df: Raw DataFrame to clean
        params: Dictionary containing all cleaning parameters from YAML
    
    Returns:
        Cleaned DataFrame ready for modeling
    """
    # Unpack YAML Sections
    dirs = params['directories']
    schema = params['schema']
    filters = params['filtering']
    cleaning = params['cleaning']
    
    # Analysis of data BEFORE cleaning 
    analyze_dataset(df, dirs['artifact_dir'])
    
    # Filter Date
    df = filter_by_date(
        df, 
        filters['min_date'], 
        filters['max_date'], 
        schema['date_column'], 
        dirs['artifact_dir']
    )
    
    # Filter Source
    df = df[df['source'] == filters['source_filter']]
    
    # Drop Columns
    df = df.drop(columns=cleaning['columns_to_drop'], errors='ignore')
    save_json(
        {'dropped': cleaning['columns_to_drop']}, 
        os.path.join(dirs['artifact_dir'], 'columns_dropped.json')
    )
    
    # Remove Empty Target Rows
    target = schema['target_col']
    df.loc[:, target].replace("", np.nan, inplace=True)
    df = df.dropna(subset=[target])
    
    # Save  Target Distribution
    dist = df[target].value_counts(normalize=True).to_dict()
    dist = {str(key): float(val) for key, val in dist.items()}
    save_json(dist, os.path.join(dirs['artifact_dir'], 'target_distribution.json'))
    
    # Handle Outliers
    df = handle_outliers(
        df, 
        numeric_cols=schema['continuous_features'], 
        threshold=cleaning['outlier_threshold'], 
        artifact_dir=dirs['artifact_dir']
    )
    
    # Impute Missing
    df = impute_missing_values(
        df, 
        numeric_cols=schema['continuous_features'], 
        cat_cols=schema['categorical_features'], 
        strategy=cleaning['imputation_strategy'],
        custom_values=cleaning['custom_imputation_values']
    )

    print("Succesfully made silver data.")
    
    return df

if __name__ == "__main__":
    params = load_params('params.yaml')
    input_path = os.path.join(".", "data", "raw_data.csv")
    df_raw = pd.read_csv(input_path)
    df_cleaned = clean_data(df_raw, params) 
    output_path = os.path.join(".", "data", "silver_data.csv")  
    df_cleaned.to_csv(output_path, index=False)