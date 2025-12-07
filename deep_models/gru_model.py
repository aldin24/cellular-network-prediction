from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam


class GRUModel:
    """GRU Model - Best RNN so far"""

    def __init__(self, input_shape):
        self.model_name = "GRU Model"

        self.model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            GRU(64),
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
