import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class MLPModel:
    """Fully Connected Neural Network"""

    def __init__(self, input_shape):
        self.model_name = "MLP Model"

        self.model = Sequential([
            Dense(128, activation="relu", input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
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
