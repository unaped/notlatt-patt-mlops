import pandas as pd

def create_dummy_cols(df, col):
    """Create one-hot encoding columns in the data."""
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def prepare_categorical_features(data, cat_cols):
    """Process categorical variables with one-hot encoding."""
    cat_vars = data[cat_cols].copy()
    
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
    
    return cat_vars

def prepare_training_data(data_path, cat_cols):
    """Load and prepare data for training."""
    data = pd.read_csv(data_path)
    print(f"Training data length: {len(data)}")
    
    # Drop unnecessary columns
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    
    # Split categorical and other variables
    other_vars = data.drop(cat_cols, axis=1)
    cat_vars = prepare_categorical_features(data, cat_cols)
    
    # Combine and convert to float
    data = pd.concat([other_vars, cat_vars], axis=1)
    for col in data:
        data[col] = data[col].astype("float64")
    
    return data