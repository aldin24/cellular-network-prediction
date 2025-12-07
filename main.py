import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import load_kaggle_dataset
from preprocess import preprocess_data

plt.style.use("ggplot")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)

# ============================
# IMPORT REGRESSION MODELS
# ============================
from regression_models.linearregression import LinearRegressionModel
from regression_models.polynomialregression import PolynomialRegressionModel
from regression_models.decisiontree import DecisionTreeModel
from regression_models.randomforest import RandomForestModel
from regression_models.xgboost_model import XGBoostModel

# ======================================================
# LOAD + PREPROCESS
# ======================================================
dataset_name = "aeryss/lte-dataset"
df = load_kaggle_dataset(dataset_name, add_source_file=True)

X, y = preprocess_data(df)
y_log = np.log1p(y)

# ======================================================
# TRAIN/TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================
# CHOOSE MODEL TO RUN
# ======================================================

# model = LinearRegressionModel()
# model = PolynomialRegressionModel(degree=3)
# model = DecisionTreeModel(max_depth=12, min_samples_split=20, min_samples_leaf=10)
# model = RandomForestModel()
model = XGBoostModel()

print(f"\nTraining model: {model.model_name}\n")

# ======================================================
# TRAIN
# ======================================================
model.train(X_train_scaled, y_train)

# ======================================================
# PREDICT
# ======================================================
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# ======================================================
# EVALUATE
# ======================================================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# ======================================================
# PLOT RESULTS
# ======================================================
plt.figure(figsize=(12, 5))
plt.plot(y_true[:300], label="True", color="crimson")
plt.plot(y_pred[:300], label="Predicted", color="gold")
plt.title(f"{model.model_name} – Prediction Sample")
plt.legend()
plt.tight_layout()
plt.show()
