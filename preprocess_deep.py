import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from preprocess_shared import (
    clean_raw_dataset,
    add_engineered_features
)


# ============================================================
#     MAIN FUNCTION FOR LSTM / GRU / CNN PREPROCESSING
# ============================================================

def preprocess_for_deep_learning(df, window_size=10):
    """
    Preprocess dataset for sequence-based deep learning models.
    Performs:

      1) Raw dataset cleaning
      2) Feature engineering (lags, mobility, distance bands)
      3) Cyclical time encoding
      4) Label encoding of categorical features
      5) Global MinMax scaling
      6) Sequence building per CellID (sliding window)

    Returns
    -------
    X_seq : (num_sequences, window_size, num_features)
    y_seq : (num_sequences,)
    scaler : fitted MinMaxScaler for inference
    """

    # --------------------------------------------------------
    # 1) CLEAN + FEATURE ENGINEERING
    # --------------------------------------------------------
    df_clean = clean_raw_dataset(df)
    df_eng = add_engineered_features(df_clean)

    # Convert timestamp for sorting & time features
    df_eng["Timestamp"] = pd.to_datetime(
        df_eng["Timestamp"], format="%Y.%m.%d_%H.%M.%S"
    )

    # Throughput target already created as DL_Mbps in feature eng.
    # No unused y_log variables anymore.

    # --------------------------------------------------------
    # 2) TIME CYCLICAL FEATURES
    # --------------------------------------------------------
    df_eng["Hour"] = df_eng["Timestamp"].dt.hour
    df_eng["DayOfWeek"] = df_eng["Timestamp"].dt.dayofweek

    df_eng["Hour_sin"] = np.sin(2 * np.pi * df_eng["Hour"] / 24)
    df_eng["Hour_cos"] = np.cos(2 * np.pi * df_eng["Hour"] / 24)
    df_eng["DoW_sin"] = np.sin(2 * np.pi * df_eng["DayOfWeek"] / 7)
    df_eng["DoW_cos"] = np.cos(2 * np.pi * df_eng["DayOfWeek"] / 7)

    # Remove raw linear time fields
    df_eng = df_eng.drop(columns=["Hour", "DayOfWeek"])

    # --------------------------------------------------------
    # 3) ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------
    nm_le = LabelEncoder()
    df_eng["NetworkMode"] = nm_le.fit_transform(df_eng["NetworkMode"])

    db_le = LabelEncoder()
    df_eng["DistBand"] = db_le.fit_transform(df_eng["DistBand"].astype(str))

    # --------------------------------------------------------
    # 4) SELECT FEATURES & SCALE
    # --------------------------------------------------------
    exclude_cols = ["Timestamp", "CellID", "DL_bitrate", "DL_Mbps"]
    feature_cols = [c for c in df_eng.columns if c not in exclude_cols]

    scaler = MinMaxScaler()
    df_eng_scaled = df_eng.copy()
    df_eng_scaled[feature_cols] = scaler.fit_transform(df_eng[feature_cols])

    # --------------------------------------------------------
    # 5) SEQUENCE CREATION PER CELLID
    # --------------------------------------------------------
    X_seq = []
    y_seq = []

    for cell_id, group in df_eng_scaled.groupby("CellID"):

        # Sort chronologically INSIDE each cell
        group = group.sort_values("Timestamp")

        X_values = group[feature_cols].values
        y_values = np.log1p(group["DL_Mbps"].values)  # Log transformed target

        for i in range(len(X_values) - window_size):
            window = X_values[i : i + window_size]
            target = y_values[i + window_size]

            # Skip bad windows
            if np.isnan(window).any() or np.isnan(target):
                continue

            X_seq.append(window)
            y_seq.append(target)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return X_seq, y_seq, scaler
