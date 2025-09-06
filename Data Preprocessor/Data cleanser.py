import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def preprocess_data(file_path, features_to_keep=None):
    """
    Preprocess the CSV file with optional feature selection
    
    Parameters:
    file_path (str): Path to the CSV file
    features_to_keep (list): List of column names to keep in the final dataset.
                             If None, all columns will be kept.
    
    Returns:
    pd.DataFrame: Preprocessed dataframe
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    
    # 1. Convert date columns to datetime
    date_columns = ['signup_date', 'order_date', 'last_order_date', 'rating_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
    
    # 2. Handle categorical variables
    categorical_cols = ['gender', 'age', 'city', 'restaurant_name', 'dish_name', 
                       'category', 'payment_method', 'delivery_status']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # 3. Convert binary columns to appropriate format
    if 'churned' in df.columns:
        df['churned'] = df['churned'].map({'Active': 0, 'Inactive': 1})
    
    # 4. Create new features
    # Calculate days since last order
    if 'last_order_date' in df.columns:
        df['days_since_last_order'] = (datetime.now() - df['last_order_date']).dt.days
    
    # Calculate customer tenure (days since signup)
    if 'signup_date' in df.columns:
        df['customer_tenure'] = (datetime.now() - df['signup_date']).dt.days
    
    # Calculate order value (price/quantity)
    if all(col in df.columns for col in ['price', 'quantity']):
        df['order_value'] = df['price'] / df['quantity']
        # Replace infinities with NaN and then fill
        df['order_value'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 5. Handle missing values if any
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 6. Select only the requested features
    if features_to_keep is not None:
        # Ensure all requested features exist in the dataframe
        available_features = [col for col in features_to_keep if col in df.columns]
        missing_features = [col for col in features_to_keep if col not in df.columns]
        
        if missing_features:
            print(f"Warning: The following requested features were not found: {missing_features}")
        
        df = df[available_features]
    
    return df

def get_encoding_methods(selected_features, df):
    """
    Ask user to select encoding method for each feature
    
    Parameters:
    selected_features (list): List of selected feature names
    df (pd.DataFrame): Dataframe with the selected features
    
    Returns:
    dict: Dictionary mapping feature names to encoding methods
    """
    encoding_methods = {}
    
    print("\nAvailable encoding methods:")
    print("1. Keep as-is (no encoding)")
    print("2. One-hot encoding")
    print("3. Frequency encoding")
    print("4. Label encoding")
    print("5. Ordinal encoding")
    print("6. Binary encoding (for binary features)")
    print("7. Drop feature")
    
    # Mapping from input to method name
    method_map = {
        '1': 'keep',
        '2': 'onehot',
        '3': 'frequency',
        '4': 'label',
        '5': 'ordinal',
        '6': 'binary',
        '7': 'drop'
    }
    
    for feature in selected_features:
        print(f"\nFeature: {feature}")
        print(f"Data type: {df[feature].dtype}")
        print(f"Unique values: {df[feature].nunique()}")
        if df[feature].nunique() <= 10:
            print(f"Sample values: {df[feature].unique()}")
        
        try:
            method_input = input("Select encoding method (1-7): ").strip()
            encoding_methods[feature] = method_map.get(method_input, 'keep')
        except:
            print("Invalid input. Using 'keep as-is' as default.")
            encoding_methods[feature] = 'keep'
    
    return encoding_methods

def apply_custom_encoding(df, encoding_methods):
    """
    Apply custom encoding methods to each feature based on user selection
    
    Parameters:
    df (pd.DataFrame): Dataframe with selected features
    encoding_methods (dict): Dictionary mapping feature names to encoding methods
    
    Returns:
    pd.DataFrame: Dataframe with encoded features
    """
    print("\nApplying custom encoding to features...")
    
    # Make a copy of the dataframe to avoid modifying the original
    encoded_df = df.copy()
    
    # Track which features to drop
    features_to_drop = []
    
    # Track one-hot encoded features to exclude from normalization
    onehot_features = []
    
    # Define encoding functions
    def apply_onehot_encoding(feature):
        if encoded_df[feature].dtype.name == 'category' or encoded_df[feature].dtype == 'object':
            dummies = pd.get_dummies(encoded_df[feature], prefix=feature, dtype=np.uint8)  # Ensure 0/1 output
            # Add the one-hot encoded columns to the dataframe
            for col in dummies.columns:
                encoded_df[col] = dummies[col]
                onehot_features.append(col)  # Track one-hot encoded columns
            return True
        else:
            print(f"Warning: {feature} is not categorical. Skipping one-hot encoding.")
            return False
    
    def apply_frequency_encoding(feature):
        if encoded_df[feature].dtype.name == 'category' or encoded_df[feature].dtype == 'object':
            freq = encoded_df[feature].value_counts(normalize=True)
            encoded_df[f"{feature}_freq"] = encoded_df[feature].map(freq)
            return True
        else:
            # For numeric features, we can still calculate frequency
            freq = encoded_df[feature].value_counts(normalize=True)
            encoded_df[f"{feature}_freq"] = encoded_df[feature].map(freq)
            return False  # Don't drop the original numeric feature by default
    
    def apply_label_encoding(feature):
        if encoded_df[feature].dtype.name == 'category' or encoded_df[feature].dtype == 'object':
            encoded_df[f"{feature}_label"] = pd.factorize(encoded_df[feature])[0]
            return True
        else:
            print(f"Warning: {feature} is not categorical. Skipping label encoding.")
            return False
    
    def apply_ordinal_encoding(feature):
        if encoded_df[feature].dtype.name == 'category' or encoded_df[feature].dtype == 'object':
            encoded_df[f"{feature}_ordinal"] = pd.factorize(encoded_df[feature])[0]
            return True
        else:
            print(f"Warning: {feature} is not categorical. Skipping ordinal encoding.")
            return False
    
    def apply_binary_encoding(feature):
        if encoded_df[feature].nunique() == 2:
            encoded_df[f"{feature}_binary"] = pd.factorize(encoded_df[feature])[0]
            return True
        else:
            print(f"Warning: {feature} is not binary ({encoded_df[feature].nunique()} unique values). Skipping binary encoding.")
            return False
    
    # Use a switch-case pattern with dictionary mapping
    encoding_switcher = {
        'keep': lambda feature: None,  # No action needed
        'onehot': lambda feature: features_to_drop.append(feature) if apply_onehot_encoding(feature) else None,
        'frequency': lambda feature: features_to_drop.append(feature) if apply_frequency_encoding(feature) else None,
        'label': lambda feature: features_to_drop.append(feature) if apply_label_encoding(feature) else None,
        'ordinal': lambda feature: features_to_drop.append(feature) if apply_ordinal_encoding(feature) else None,
        'binary': lambda feature: features_to_drop.append(feature) if apply_binary_encoding(feature) else None,
        'drop': lambda feature: features_to_drop.append(feature)
    }
    
    for feature, method in encoding_methods.items():
        if feature not in encoded_df.columns:
            continue
            
        print(f"Processing {feature} with {method} encoding")
        
        # Execute the appropriate encoding function
        encoding_switcher.get(method, lambda feature: None)(feature)
    
    # Drop features that were marked for removal
    if features_to_drop:
        encoded_df.drop(features_to_drop, axis=1, inplace=True)
    
    print(f"Data shape after encoding: {encoded_df.shape}")
    
    # Store one-hot features as an attribute for later reference
    encoded_df.onehot_features = onehot_features
    
    return encoded_df

def get_normalization_methods(encoded_df):
    """
    Ask user to select normalization method for each numeric feature
    
    Parameters:
    encoded_df (pd.DataFrame): Dataframe with encoded features
    
    Returns:
    dict: Dictionary mapping feature names to normalization methods
    """
    normalization_methods = {}
    
    # Get numeric features only (excluding binary/categorical)
    numeric_features = encoded_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove binary features (0/1 values) and one-hot encoded features
    binary_features = []
    for feature in numeric_features:
        unique_vals = encoded_df[feature].unique()
        if set(unique_vals).issubset({0, 1}) and len(unique_vals) <= 2:
            binary_features.append(feature)
    
    # Also exclude one-hot encoded features from normalization
    onehot_features = getattr(encoded_df, 'onehot_features', [])
    
    numeric_features = [f for f in numeric_features 
                       if f not in binary_features and f not in onehot_features]
    
    if not numeric_features:
        print("No numeric features available for normalization.")
        return normalization_methods
    
    print("\nAvailable normalization methods:")
    print("1. No normalization")
    print("2. Standard scaling (mean=0, std=1)")
    print("3. Min-Max scaling (range 0-1)")
    print("4. Robust scaling (resistant to outliers)")
    
    # Mapping from input to method name
    method_map = {
        '1': 'none',
        '2': 'standard',
        '3': 'minmax',
        '4': 'robust'
    }
    
    print(f"\nNumeric features available for normalization:")
    for i, feature in enumerate(numeric_features, 1):
        print(f"{i}. {feature} (range: {encoded_df[feature].min():.2f} to {encoded_df[feature].max():.2f})")
    
    for feature in numeric_features:
        print(f"\nFeature: {feature}")
        print(f"Current range: {encoded_df[feature].min():.2f} to {encoded_df[feature].max():.2f}")
        print(f"Mean: {encoded_df[feature].mean():.2f}, Std: {encoded_df[feature].std():.2f}")
        
        try:
            method_input = input("Select normalization method (1-4): ").strip()
            normalization_methods[feature] = method_map.get(method_input, 'none')
        except:
            print("Invalid input. Using 'no normalization' as default.")
            normalization_methods[feature] = 'none'
    
    return normalization_methods

def apply_normalization(encoded_df, normalization_methods):
    """
    Apply normalization methods to each feature based on user selection
    
    Parameters:
    encoded_df (pd.DataFrame): Dataframe with encoded features
    normalization_methods (dict): Dictionary mapping feature names to normalization methods
    
    Returns:
    pd.DataFrame: Dataframe with normalized features
    """
    print("\nApplying normalization to features...")
    
    # Make a copy of the dataframe to avoid modifying the original
    normalized_df = encoded_df.copy()
    
    # Initialize scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    # Track which features were normalized
    normalized_features = []
    
    for feature, method in normalization_methods.items():
        if feature not in normalized_df.columns:
            continue
            
        print(f"Processing {feature} with {method} normalization")
        
        if method == 'none':
            continue  # No normalization needed
            
        elif method == 'standard':
            normalized_df[feature] = standard_scaler.fit_transform(normalized_df[[feature]])
            normalized_features.append(feature)
            
        elif method == 'minmax':
            normalized_df[feature] = minmax_scaler.fit_transform(normalized_df[[feature]])
            normalized_features.append(feature)
            
        elif method == 'robust':
            normalized_df[feature] = robust_scaler.fit_transform(normalized_df[[feature]])
            normalized_features.append(feature)
    
    if normalized_features:
        print(f"\nNormalized features: {normalized_features}")
        for feature in normalized_features:
            print(f"{feature}: range {normalized_df[feature].min():.2f} to {normalized_df[feature].max():.2f}")
    
    return normalized_df

def get_feature_selection():
    """
    Interactive function to get feature selection from user
    """
    print("Available features in the dataset:")
    print("1. customer_id")
    print("2. gender")
    print("3. age")
    print("4. city")
    print("5. signup_date")
    print("6. order_id")
    print("7. order_date")
    print("8. restaurant_name")
    print("9. dish_name")
    print("10. category")
    print("11. quantity")
    print("12. price")
    print("13. payment_method")
    print("14. order_frequency")
    print("15. last_order_date")
    print("16. loyalty_points")
    print("17. churned")
    print("18. rating")
    print("19. rating_date")
    print("20. delivery_status")
    print("21. days_since_last_order (derived)")
    print("22. customer_tenure (derived)")
    print("23. order_value (derived)")
    print("\nEnter the numbers of features you want to keep (comma-separated).")
    print("For example: 1,2,3,4,17,18,21,22,23")
    print("Or press Enter to keep all features.")
    
    selection = input("Your selection: ").strip()
    
    if not selection:
        return None
    
    # Map numbers to feature names using dictionary (switch-case pattern)
    feature_map = {
        '1': 'customer_id',
        '2': 'gender',
        '3': 'age',
        '4': 'city',
        '5': 'signup_date',
        '6': 'order_id',
        '7': 'order_date',
        '8': 'restaurant_name',
        '9': 'dish_name',
        '10': 'category',
        '11': 'quantity',
        '12': 'price',
        '13': 'payment_method',
        '14': 'order_frequency',
        '15': 'last_order_date',
        '16': 'loyalty_points',
        '17': 'churned',
        '18': 'rating',
        '19': 'rating_date',
        '20': 'delivery_status',
        '21': 'days_since_last_order',
        '22': 'customer_tenure',
        '23': 'order_value'
    }
    
    try:
        selected_numbers = [num.strip() for num in selection.split(',')]
        selected_features = [feature_map.get(num, f'unknown_{num}') for num in selected_numbers]
        
        # Remove any unknown features
        selected_features = [f for f in selected_features if not f.startswith('unknown_')]
        
        return selected_features
    except Exception as e:
        print(f"Error: {e}. Using all features as default.")
        return None

# Execute the preprocessing
if __name__ == "__main__":
    # Load the raw data first to display a sample
    raw_df = pd.read_csv('Raw_Data.csv')
    print("="*60)
    print("RAW DATA SAMPLE (BEFORE PROCESSING)")
    print("="*60)
    print(f"Raw data shape: {raw_df.shape}")
    print("\nFirst 5 rows of raw data:")
    print(raw_df.head())
    print("\nData types of raw data:")
    print(raw_df.dtypes)
    print("="*60)
    
    # Get feature selection from user
    selected_features = get_feature_selection()
    
    # Load and preprocess the data with selected features
    processed_df = preprocess_data('Raw_Data.csv', selected_features)
    
    print(f"\nData shape after initial preprocessing: {processed_df.shape}")
    print("Data types after preprocessing:")
    print(processed_df.dtypes)
    
    # Display sample of preprocessed data (before encoding)
    print("\nSample of preprocessed data (before encoding):")
    print(processed_df.head())
    
    # Get encoding methods from user for each selected feature
    encoding_methods = get_encoding_methods(selected_features, processed_df)
    
    # Apply custom encoding based on user selection
    encoded_df = apply_custom_encoding(processed_df, encoding_methods)
    
    # Display the encoded dataframe info
    print("\nData after encoding:")
    print(encoded_df.info())
    
    # Display sample of encoded data (before normalization)
    print("\nSample of encoded data (before normalization):")
    print(encoded_df.head())
    
    # Get normalization methods from user for each numeric feature
    normalization_methods = get_normalization_methods(encoded_df)
    
    # Apply normalization based on user selection
    if normalization_methods:
        final_df = apply_normalization(encoded_df, normalization_methods)
    else:
        final_df = encoded_df
    
    # Display the final dataframe info
    print("\nFinal data info after all processing:")
    print(final_df.info())
    
    # Save the processed data
    output_filename = 'Fully_Preprocessed_Data.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\nPreprocessed and encoded data saved as '{output_filename}'")
    
    # Display basic info about the processed data
    print("\nSample of final processed data:")
    print(final_df.head())
    
    # Show comparison between raw and processed data
    print("="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Original data shape: {raw_df.shape}")
    print(f"Final processed data shape: {final_df.shape}")
    print(f"Number of features reduced by: {raw_df.shape[1] - final_df.shape[1]}")
    print("="*60)