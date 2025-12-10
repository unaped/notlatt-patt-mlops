import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pprint import pprint

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)


def describe_numeric_col(x):
    """Calculate descriptive statistics for a numeric column."""
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )


def impute_missing_values(x, method="mean"):
    """Impute missing values using mean/median for numeric or mode for categorical."""
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x


def setup_artifacts_directory(artifacts_dir="./models"):
    """Create artifacts directory if it doesn't exist."""
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"Created artifacts directory: {artifacts_dir}")


def load_raw_data(data_path="./data/raw/raw_data.csv"):
    """Load raw data from CSV file."""
    print("Loading training data")
    data = pd.read_csv(data_path)
    print(f"Total rows: {len(data)}")
    return data


def filter_by_date_range(data, min_date, max_date):
    """Filter data by date range and save date limits."""
    if not max_date:
        max_date = pd.to_datetime(pd.Timestamp.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()
    
    min_date = pd.to_datetime(min_date).date()
    
    # Filter by date range
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    
    # Save date limits
    actual_min = data["date_part"].min()
    actual_max = data["date_part"].max()
    date_limits = {"min_date": str(actual_min), "max_date": str(actual_max)}
    
    with open("./data/interim/date_limits.json", "w") as f:
        json.dump(date_limits, f)
    
    print(f"Filtered data from {actual_min} to {actual_max}")
    return data


def select_features(data):
    """Remove irrelevant columns from dataset."""
    # Drop irrelevant columns
    data = data.drop(
        [
            "is_active", "marketing_consent", "first_booking", 
            "existing_customer", "last_seen"
        ],
        axis=1
    )
    
    # Drop columns that will be added back after EDA
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1
    )
    
    return data


def clean_data(data):
    """Clean data by removing invalid rows."""
    # Replace empty strings with NaN
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    
    # Drop rows with missing critical values
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    
    # Filter by source
    data = data[data.source == "signup"]
    
    # Print target distribution
    result = data.lead_indicator.value_counts(normalize=True)
    print("\nTarget value distribution:")
    for val, n in zip(result.index, result):
        print(f"{val}: {n:.3f}")
    
    return data


def convert_to_categorical(data):
    """Convert specified columns to categorical type."""
    cat_columns = [
        "lead_id", "lead_indicator", "customer_group", 
        "onboarding", "source", "customer_code"
    ]
    
    for col in cat_columns:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")
    
    return data


def separate_variable_types(data):
    """Separate continuous and categorical variables."""
    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]
    
    print("\nContinuous columns:")
    pprint(list(cont_vars.columns), indent=4)
    print("\nCategorical columns:")
    pprint(list(cat_vars.columns), indent=4)
    
    return cont_vars, cat_vars


def remove_outliers(cont_vars):
    """Remove outliers using Z-score method (2 standard deviations)."""
    cont_vars = cont_vars.apply(
        lambda x: x.clip(
            lower=(x.mean() - 2 * x.std()),
            upper=(x.mean() + 2 * x.std())
        )
    )
    
    # Save outlier summary
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv('./data/interim/outlier_summary.csv')
    
    return cont_vars


def impute_data(cont_vars, cat_vars):
    """Impute missing values for both continuous and categorical variables."""
    # Save categorical imputation values
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv("./data/interim/cat_missing_impute.csv")
    
    # Impute continuous variables
    cont_vars = cont_vars.apply(impute_missing_values)
    
    # Impute categorical variables
    cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)
    
    print("\nMissing values after imputation:")
    print(cat_vars.apply(
        lambda x: pd.Series([x.count(), x.isnull().sum()], 
        index=['Count', 'Missing'])
    ).T)
    
    return cont_vars, cat_vars


def standardize_data(cont_vars, scaler_path="./models/scaler.pkl"):
    """Standardize continuous variables using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    
    # Save scaler for later use
    joblib.dump(value=scaler, filename=scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    cont_vars = pd.DataFrame(
        scaler.transform(cont_vars), 
        columns=cont_vars.columns
    )
    
    return cont_vars


def combine_data(cont_vars, cat_vars):
    """Combine continuous and categorical variables."""
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    
    print(f"\nData cleansed and combined. Rows: {len(data)}")
    return data


def save_data_drift_artifact(data):
    """Save column names for data drift monitoring."""
    data_columns = list(data.columns)
    with open('./data/interim/columns_drift.json', 'w+') as f:
        json.dump(data_columns, f)
    
    data.to_csv('./data/interim/training_data.csv', index=False)
    print("Saved training data and drift columns")


def bin_source_column(data):
    """Create binned source column for grouping."""
    data['bin_source'] = data['source']
    
    mapping = {
        'li': 'socials',
        'fb': 'socials',
        'organic': 'group1',
        'signup': 'group1'
    }
    
    data['bin_source'] = data['source'].map(mapping)
    return data


def save_gold_dataset(data, output_path='./data/processed/train_data_gold.csv'):
    """Save the final processed dataset."""
    data.to_csv(output_path, index=False)
    print(f"Saved gold dataset to {output_path}")


def preprocess_pipeline(min_date="2024-01-01", max_date="2024-01-31"):
    """Execute the complete preprocessing pipeline."""
    # Setup
    setup_artifacts_directory()
    
    # Load data
    data = load_raw_data()
    
    # Filter by date
    data = filter_by_date_range(data, min_date, max_date)
    
    # Feature selection
    data = select_features(data)
    
    # Clean data
    data = clean_data(data)
    
    # Convert types
    data = convert_to_categorical(data)
    
    # Separate variable types
    cont_vars, cat_vars = separate_variable_types(data)
    
    # Remove outliers
    cont_vars = remove_outliers(cont_vars)
    
    # Impute missing values
    cont_vars, cat_vars = impute_data(cont_vars, cat_vars)
    
    # Standardize
    cont_vars = standardize_data(cont_vars)
    
    # Combine
    data = combine_data(cont_vars, cat_vars)
    
    # Save artifacts
    save_data_drift_artifact(data)
    
    # Bin source column
    data = bin_source_column(data)
    
    # Save final dataset
    save_gold_dataset(data)
    
    print("\n=== Preprocessing Complete ===")
    return data


if __name__ == "__main__":
    preprocess_pipeline()