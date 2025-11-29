import kagglehub
import os
import pandas as pd
import numpy as np

# Download latest version
path = kagglehub.dataset_download("suraj520/cellular-network-analysis-dataset")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith(".csv")]

# Load the CSV file
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)

# Data investigation and preprocessing
df_clean = df.copy()
cols_to_remove = [
    'Locality',  # String/categorical
    'Signal Quality (%)',  # All zeros (no variance)
    'srsRAN Measurement (dBm)',  # Highly correlated with others
    'BladeRFxA9 Measurement (dBm)'  # Highly correlated with others
]

cols_actually_removed = [col for col in cols_to_remove if col in df_clean.columns]
df_clean = df_clean.drop(columns=cols_actually_removed, errors='ignore')

print("\nCleaned dataframe info:")
print(df_clean.info())
