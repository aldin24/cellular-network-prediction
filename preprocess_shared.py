import pandas as pd
import numpy as np

def clean_raw_dataset(df):
    df = df.copy()

    # Keep only records where download state = active
    df = df[df["State"] == "D"]

    # Remove columns that cannot help with prediction
    drop_cols = ["Operatorname", "UL_bitrate", "source_file", "State"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Columns that must be converted from string → numeric
    numeric_cols = [
        "RSRQ", "SNR", "CQI", "RSSI",
        "NRxRSRP", "NRxRSRQ",
        "ServingCell_Lon", "ServingCell_Lat", "ServingCell_Distance"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix impossible RSRQ values (RSRQ must be <= -3)
    df.loc[df["RSRQ"] > -2, "RSRQ"] = np.nan

    # Fix impossible values for distance (must be < 50 km)
    df = df[df["ServingCell_Distance"] < 50000]

    return df



def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic temporal + mobility features to the cleaned dataset.

    Assumes:
      - 'Timestamp' is still present and string formatted "%Y.%m.%d_%H.%M.%S"
      - 'CellID', 'Speed', 'ServingCell_Distance', 'DL_bitrate' exist
    """

    df = df.copy()

    # 1) Ensure datetime + sort by cell & time
    df["Timestamp"] = pd.to_datetime(df["Timestamp"],format="%Y.%m.%d_%H.%M.%S")
    df = df.sort_values(["CellID", "Timestamp"])

    # 2) Target in Mbps (we'll still use DL_bitrate as label later,
    #    but this column is handy for lag features)
    df["DL_Mbps"] = df["DL_bitrate"] / 1000.0

    # 3) Lag features (per cell)
    for lag in [1, 2, 5]:
        df[f"DL_Mbps_lag{lag}"] = (
            df.groupby("CellID")["DL_Mbps"].shift(lag)
        )

    # 4) Mobility / handover features
    # speed change between consecutive samples in the same cell
    df["Speed_delta"] = (
        df.groupby("CellID")["Speed"].diff()
    )

    # whether UE changed serving cell compared to previous sample
    prev_cell = df.groupby("CellID")["CellID"].shift(1)
    df["HandoverFlag"] = (df["CellID"] != prev_cell).astype(int)

    # rough "fast vs slow" indicator (e.g. > 30 km/h)
    df["IsFast"] = (df["Speed"] >= 30).astype(int)

    # 5) Distance bands (very light spatial feature)
    dist = df["ServingCell_Distance"].astype(float)
    bins = [0, 100, 500, 1000, 5000, np.inf]
    labels = ["0-100", "100-500", "500-1k", "1k-5k", ">5k"]
    df["DistBand"] = pd.cut(dist, bins=bins, labels=labels,
                            include_lowest=True)

    # 6) Drop first few rows in each cell where we don't have lags
    lag_cols = [f"DL_Mbps_lag{lag}" for lag in [1, 2, 5]]
    df = df.dropna(subset=lag_cols + ["Speed_delta"])

    return df