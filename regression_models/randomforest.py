from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


class RandomForestModel:
    """Random Forest Regression Model"""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.model = RandomForestRegressor(
            n_estimators=300,  # More trees = more stability
            max_depth=25,  # Prevents overly complex trees
            min_samples_split=50,  # Avoid splits on noisy samples
            min_samples_leaf=20,  # Avoid tiny leaves
            max_features="sqrt",  # Improves generalization
            bootstrap=True,  # Standard for RF
            n_jobs=-1,  # Use all CPU cores
            random_state=42
        )
        self.model_name = "Random Forest Regression"
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

