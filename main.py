import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from linearregression import LinearRegressionModel

# Download and load data
path = kagglehub.dataset_download("suraj520/cellular-network-analysis-dataset")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith(".csv")]
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)

# Preprocess data
print("Preprocessing data...")
X, y = preprocess_data(df)

print(f"\nPreprocessed X shape: {X.shape}")
print(f"X columns: {list(X.columns)}")
print(f"y shape: {y.shape}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================================
# SELECT MODEL TO RUN
# ========================================

# Model 1: Linear Regression
model = LinearRegressionModel()
