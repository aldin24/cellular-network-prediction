from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam


class CNNModel:
    def __init__(self, input_shape):
        from keras.layers import (
            Input, Conv1D, BatchNormalization, ReLU, Add,
            MaxPooling1D, Concatenate, GlobalAveragePooling1D,
            Dense, Dropout
        )
        from keras.models import Model
        from keras.optimizers import Adam
        from keras import regularizers

        self.model_name = "CNN Model"

        inputs = Input(shape=input_shape)

        # ============ Inception-style multi-scale block ============
        b1 = Conv1D(64, 3, padding="same",
                    kernel_regularizer=regularizers.l2(1e-4))(inputs)
        b1 = BatchNormalization()(b1)
        b1 = ReLU()(b1)

        b2 = Conv1D(64, 5, padding="same",
                    kernel_regularizer=regularizers.l2(1e-4))(inputs)
        b2 = BatchNormalization()(b2)
        b2 = ReLU()(b2)

        b3 = Conv1D(64, 7, padding="same",
                    kernel_regularizer=regularizers.l2(1e-4))(inputs)
        b3 = BatchNormalization()(b3)
        b3 = ReLU()(b3)

        x = Concatenate()([b1, b2, b3])
        x = MaxPooling1D(pool_size=2)(x)

        # ============ Deeper CNN block ============
        x = Conv1D(128, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv1D(128, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPooling1D(pool_size=2)(x)

        # ============ Dilated conv block ============
        d1 = Conv1D(256, 3, padding="same", dilation_rate=2)(x)
        d1 = BatchNormalization()(d1)
        d1 = ReLU()(d1)

        d2 = Conv1D(256, 3, padding="same", dilation_rate=4)(x)
        d2 = BatchNormalization()(d2)
        d2 = ReLU()(d2)

        x = Add()([d1, d2])    # residual-like merge
        x = ReLU()(x)

        # ============ Global pooling ============
        x = GlobalAveragePooling1D()(x)

        # ============ Dense layers ============
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)

        outputs = Dense(1)(x)

        # Build model
        self.model = Model(inputs, outputs)

        self.model.compile(
            optimizer=Adam(1e-3),
            loss="mse",
            metrics=["mae"]
        )

    def train(self, X_train, y_train, epochs=30, batch_size=64):
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()
