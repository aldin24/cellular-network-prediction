"""
Linear Regression Model
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


class LinearRegressionModel:
    """Multiple Linear Regression Model"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.model_name = "Multiple Linear Regression"
