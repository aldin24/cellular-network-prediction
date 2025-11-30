import pandas as pd
def preprocess_data(df):
    df_clean = df.copy()
    
    # Step 1: Remove unwanted columns
    cols_to_remove = [
        'Locality',  # String/categorical
        'Signal Quality (%)',  # All zeros (no variance)
        'srsRAN Measurement (dBm)',  # Highly correlated with others
        'BladeRFxA9 Measurement (dBm)'  # Highly correlated with others
    ]
    
    cols_actually_removed = [col for col in cols_to_remove if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_actually_removed, errors='ignore')
    
    # Step 2: Split features and target
    X = df_clean.drop(columns=['Data Throughput (Mbps)'])
    y = df_clean['Data Throughput (Mbps)']
    
    # Step 3: Extract time features from Timestamp
    if 'Timestamp' in X.columns:
        X['Timestamp'] = pd.to_datetime(X['Timestamp'])
        X['Year'] = X['Timestamp'].dt.year
        X['Month'] = X['Timestamp'].dt.month
        X['Day'] = X['Timestamp'].dt.day
        X['Hour'] = X['Timestamp'].dt.hour
        X['DayOfWeek'] = X['Timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        X = X.drop(columns=['Timestamp'])
    
    # Step 4: Encode categorical variables
    if 'Network Type' in X.columns:
        X = pd.get_dummies(X, columns=['Network Type'], drop_first=True)
    
    return X, y

