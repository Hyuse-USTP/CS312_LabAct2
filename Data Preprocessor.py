import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def create_sample_data():
    """
    Creates a sample dataset with mixed data types, missing values, and outliers.
    Returns:
        pd.DataFrame: A sample dataframe with realistic data issues.
    """
    np.random.seed(42)
    sample_size = 300
    
    data = {
        'customer_id': range(sample_size),
        'age': np.random.normal(45, 15, sample_size),
        'income': np.random.lognormal(4.5, 0.6, sample_size),
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany', None], 
                                  sample_size, p=[0.35, 0.3, 0.2, 0.1, 0.05]),
        'subscription_type': np.random.choice(['Free', 'Basic', 'Premium', None], 
                                            sample_size, p=[0.5, 0.3, 0.15, 0.05]),
        'website_visits': np.random.poisson(10, sample_size),
        'purchase_amount': np.random.exponential(50, sample_size),
        'target': np.random.choice([0, 1], sample_size, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    df.loc[10, 'age'] = 200    # Outlier
    df.loc[20, 'income'] = 0   # Invalid value
    df.loc[30:35, 'income'] = np.nan  # Block of missing values
    df.loc[40:45, 'age'] = np.nan     # Block of missing values
    
    # Add random missing values
    for col in ['income', 'age', 'country']:
        df[col] = df[col].mask(np.random.random(sample_size) < 0.07, other=np.nan)
    
    # Add duplicate rows
    duplicates = df.sample(5, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

def explore_data(df):
    """
    Performs initial data exploration.
    Args:
        df (pd.DataFrame): Input dataframe to explore
    """
    print("="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nBasic Statistics for Numerical Columns:")
    print(df.describe(include=[np.number]))
    
    print("\nDuplicate Rows:", df.duplicated().sum())

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits data into training and test sets.
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of data for test set
        random_state (int): Random seed for reproducibility
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\n" + "="*50)
    print("STEP 1: TRAIN-TEST SPLIT")
    print("="*50)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution in train: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test

def get_feature_types(X_train):
    """
    Identifies numerical and categorical features automatically.
    Args:
        X_train (pd.DataFrame): Training features
    Returns:
        tuple: Lists of numerical and categorical column names
    """
    # Remove ID/date columns from features
    drop_cols = []
    for col in X_train.columns:
        if 'id' in col.lower() or 'date' in col.lower():
            drop_cols.append(col)
    X_train = X_train.drop(columns=drop_cols)
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical features: {numerical_cols}")
    print(f"Categorical features: {categorical_cols}")
    
    return numerical_cols, categorical_cols, drop_cols

def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Creates a preprocessing pipeline for numerical and categorical features.
    Args:
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print("\n" + "="*50)
    print("STEP 2-4: CREATING PREPROCESSING PIPELINE")
    print("="*50)
    
    # Numerical preprocessing: Impute missing values with median and scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: Impute missing values with mode and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop columns not specified in the transformers
    )
    
    print("Preprocessing pipeline created successfully!")
    return preprocessor

def apply_preprocessing(preprocessor, X_train, X_test):
    """
    Applies the preprocessing pipeline to training and test data.
    Args:
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
    Returns:
        tuple: Processed training and test arrays and feature names
    """
    print("\n" + "="*50)
    print("APPLYING PREPROCESSING")
    print("="*50)
    
    # Fit on training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Numerical feature names (remain unchanged)
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    feature_names.extend(num_features)
    
    # Categorical feature names (from one-hot encoding)
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
    feature_names.extend(cat_features)
    
    print(f"Processed training set shape: {X_train_processed.shape}")
    print(f"Processed test set shape: {X_test_processed.shape}")
    print(f"Number of features after preprocessing: {len(feature_names)}")
    
    return X_train_processed, X_test_processed, feature_names

def load_data_from_file(filepath):
    """
    Loads data from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    print(f"Loading data from {filepath} ...")
    # Try to parse date columns automatically
    df = pd.read_csv(filepath, parse_dates=True, infer_datetime_format=True, low_memory=False)
    print("Data loaded successfully!")
    return df

def main():
    """
    Main function to run the complete preprocessing pipeline.
    """
    print("MACHINE LEARNING DATA PREPROCESSING PIPELINE")
    print("="*50)
    
    # Step 0: Load data from file (ask user for file path first)
    file_path = input("Enter the path to your CSV data file: ").strip()
    if not os.path.exists(file_path):
        print("File not found!")
        return
    df = load_data_from_file(file_path)
    
    # Display raw data
    print("\n" + "="*50)
    print("RAW DATA (First 5 rows)")
    print("="*50)
    print(df.head())
    
    # Step 1: Explore data
    explore_data(df)
    
    # Step 2: Ask user for target column
    print("\nAvailable columns:", list(df.columns))
    target_column = input("Enter the name of the target column (e.g., 'churned'): ").strip()
    if target_column not in df.columns:
        print("Target column not found!")
        return
    
    # Step 3: Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 20 else None
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target distribution in train: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    
    # Step 4: Identify feature types and drop ID/date columns
    numerical_cols, categorical_cols, drop_cols = get_feature_types(X_train)
    # Drop ID/date columns from both train and test
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)
    
    # Step 5: Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    
    # Step 6: Apply preprocessing
    X_train_processed, X_test_processed, feature_names = apply_preprocessing(
        preprocessor, X_train, X_test
    )
    
    # Optional: Convert back to DataFrame for inspection
    train_df_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    test_df_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Display preprocessed data
    print("\n" + "="*50)
    print("PREPROCESSED TRAINING DATA (First 5 rows)")
    print("="*50)
    print(train_df_processed.head())
    
    print("\nProcessed Training Data Overview:")
    print(f"Shape: {train_df_processed.shape}")
    print("\nProcessed Test Data Overview:")
    print(f"Shape: {test_df_processed.shape}")
    
    print("\n" + "="*50)
    print("DATA IS NOW READY FOR MODEL TRAINING!")
    print("="*50)
    
    # Return processed data for further use
    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }

if __name__ == "__main__":
    # Run the complete pipeline
    processed_data = main()
    
    # The processed_data dictionary contains everything needed for modeling:
    # processed_data['X_train'] - Processed training features
    # processed_data['X_test']  - Processed test features  
    # processed_data['y_train'] - Training labels
    # processed_data['y_test']  - Test labels
    # processed_data['feature_names'] - Names of processed features
    # processed_data['preprocessor'] - Fitted preprocessor for new data