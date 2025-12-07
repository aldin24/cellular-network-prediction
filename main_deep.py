import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import seaborn as sns
from data_loader import load_kaggle_dataset
from preprocess_deep import preprocess_for_deep_learning

plt.style.use("ggplot")
sns.set_palette("husl")

# IMPORT MODELS
from deep_models.gru_model import GRUModel
from deep_models.cnn_model import CNNModel
from deep_models.lstm_model import LSTMModel
from deep_models.mlp_model import MLPModel
from deep_models.cnn_lstm_model import CNNLSTMModel

# =============================
# Load + preprocess
# =============================
df = load_kaggle_dataset("aeryss/lte-dataset")
X_seq, y_seq, scaler = preprocess_for_deep_learning(df, window_size=20)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# =============================
# Choose model here
# =============================
input_shape = (X_train.shape[1], X_train.shape[2])

# model = GRUModel(input_shape)
model = CNNModel(input_shape)
# model = LSTMModel(input_shape)
# model = MLPModel(input_shape[0] * input_shape[1])
# model = CNNLSTMModel(input_shape)

model.train(X_train, y_train, epochs=40, batch_size=64)

# =============================
# Predict
# =============================
y_pred_log = model.predict(X_test)
y_true = np.expm1(y_test)
y_pred = np.expm1(y_pred_log)

# =============================
# Evaluate
# =============================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\n==== {model.model_name} ====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
print("===========================\n")

# =============================
# Plot
# =============================
plt.figure(figsize=(14, 5))
plt.plot(y_true[:500], label="True", color="red")
plt.plot(y_pred[:500], label="Pred", color="gold")
plt.title(model.model_name)
plt.legend()
plt.show()
