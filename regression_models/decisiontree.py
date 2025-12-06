from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


class DecisionTreeModel:
    """Decision Tree Regression Model"""
    
    def __init__(self):
        self.model = DecisionTreeRegressor(
            max_depth=25,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42
        )
        self.model_name = "Decision Tree Regression"
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

