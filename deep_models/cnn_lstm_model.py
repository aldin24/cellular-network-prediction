from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam


class CNNLSTMModel:
    """Hybrid CNN → LSTM Model"""

    def __init__(self, input_shape):
        self.model_name = "CNN + LSTM Model"

        self.model = Sequential([
            Conv1D(64, 3, activation="relu", padding="same", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),

            Conv1D(128, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),

            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(1)
        ])

        self.model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])

    def train(self, X_train, y_train, epochs=30, batch_size=64):
        self.model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.2,
                       verbose=1)

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()
