import pandas as pd
from preprocess_shared import clean_raw_dataset

def preprocess_for_regression(df):
    df_clean = clean_raw_dataset(df)

    # Target in Mbps
    y = df_clean["DL_bitrate"] / 1000.0
    X = df_clean.drop(columns=["DL_bitrate"], errors="ignore")

    # Convert timestamp
    X["Timestamp"] = pd.to_datetime(X["Timestamp"], format="%Y.%m.%d_%H.%M.%S")
    X["Day"] = X["Timestamp"].dt.day
    X["Hour"] = X["Timestamp"].dt.hour
    X["DayOfWeek"] = X["Timestamp"].dt.dayofweek
    X = X.drop(columns=["Timestamp"])

    # Dummy-encode categorical NetworkMode
    X = pd.get_dummies(X, columns=["NetworkMode"], drop_first=False)
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    return X, y
