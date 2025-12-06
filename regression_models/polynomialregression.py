from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np


class PolynomialRegressionModel:
    """Polynomial Regression Model"""
    
    def __init__(self, degree=2):
        self.degree = degree
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        self.model_name = f"Polynomial Regression (degree={degree})"
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

