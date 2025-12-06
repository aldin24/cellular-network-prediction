import numpy as np
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_loader import load_kaggle_dataset
from preprocess import preprocess_data
plt.style.use('ggplot')
sns.set_palette("husl")
import pandas as pd
pd.set_option('display.max_columns', None)

from regression_models import LinearRegressionModel, RandomForestModel, XGBoostModel, DecisionTreeModel

# Download and load data
dataset_name = "aeryss/lte-dataset"
df = load_kaggle_dataset(dataset_name, add_source_file=True)

X, y = preprocess_data(df)
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y_log, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train model
model = XGBoostModel()
model.train(X_train_scaled, y_train)

# Predictions (in log-space)
y_pred_log = model.predict(X_test_scaled)

# Convert back to Mbps
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)