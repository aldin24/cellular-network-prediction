from xgboost import XGBRegressor
import numpy as np


class XGBoostModel:
    """XGBoost Regression Model"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42):
        self.model = XGBRegressor(
            n_estimators=600,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=1,
            reg_lambda=1,
            gamma=0,
            n_jobs=-1,
            random_state=42,
            tree_method="hist"
        )
        self.model_name = "XGBoost Regression"
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

