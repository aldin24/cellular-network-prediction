import pandas as pd
import numpy as np

def preprocess_data(df):
    df_clean = df.copy()

    df_clean = df_clean[df_clean["State"] == "D"]
    cols_to_drop = [
        "Operatorname",
        "CellID",
        "UL_bitrate",
        "source_file",
        "State",
    ]
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    numeric_cols = [
        "RSRQ", "SNR", "CQI", "RSSI",
        "NRxRSRP", "NRxRSRQ",
        "ServingCell_Lon", "ServingCell_Lat", "ServingCell_Distance"
    ]

    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # clean impossible values
    df_clean["RSRQ"] = df_clean["RSRQ"].astype(float)
    df_clean.loc[df_clean["RSRQ"] > -2, "RSRQ"] = np.nan
    df_clean["ServingCell_Distance"] = df_clean["ServingCell_Distance"].astype(float)
    df_clean = df_clean[df_clean["ServingCell_Distance"] < 50000]  # remove bad values

    y = df_clean['DL_bitrate'] / 1000.0  # Convert kbit/s to Mbps
    X = df_clean.drop(columns=['DL_bitrate'], errors='ignore')

    # extract time features from Timestamp
    X['Timestamp'] = pd.to_datetime(X['Timestamp'], format="%Y.%m.%d_%H.%M.%S")
    X['Day'] = X['Timestamp'].dt.day
    X['Hour'] = X['Timestamp'].dt.hour
    X['DayOfWeek'] = X['Timestamp'].dt.dayofweek
    X = X.drop(columns=['Timestamp'])

    X = pd.get_dummies(X, columns=["NetworkMode"], drop_first=False)

    dummy_cols = [col for col in X.columns if col.startswith("NetworkMode_")]
    X[dummy_cols] = X[dummy_cols].astype(int)

    return X, y
