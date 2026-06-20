import os
import sys
from dataclasses import dataclass

from scipy.stats import randint, uniform
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.config.paths import MODEL_PATH
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = MODEL_PATH


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "RandomForest Regressor": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGB Regressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoost Regressor": CatBoostRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
            }
            # for hyper param tuning
            param_distributions = {
                "Linear Regression": {
                    "fit_intercept": [True, False]
                },

                "KNeighbors Regressor": {
                    "n_neighbors": randint(2, 30),
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },

                "Decision Tree": {
                    "max_depth": [None] + list(range(3, 21)),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["sqrt", "log2", None]
                },

                "RandomForest Regressor": {
                    "n_estimators": randint(100, 1000),
                    "max_depth": [None] + list(range(5, 30)),
                    "min_samples_split": randint(2, 20),
                    "min_samples_leaf": randint(1, 10),
                    "max_features": ["sqrt", "log2", None]
                },

                "Gradient Boosting": {
                    "n_estimators": randint(100, 1000),
                    "learning_rate": uniform(0.01, 0.3),
                    "max_depth": randint(3, 10),
                    "subsample": uniform(0.6, 0.4),
                    "min_samples_split": randint(2, 20)
                },

                "AdaBoost Regressor": {
                    "n_estimators": randint(50, 500),
                    "learning_rate": uniform(0.01, 1.0),
                    "loss": ["linear", "square", "exponential"]
                },

                "XGB Regressor": {
                    "n_estimators": randint(100, 1000),
                    "max_depth": randint(3, 10),
                    "learning_rate": uniform(0.01, 0.3),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                    "min_child_weight": randint(1, 10)
                },

                "CatBoost Regressor": {
                    "iterations": randint(200, 1000),
                    "depth": randint(4, 10),
                    "learning_rate": uniform(0.01, 0.3),
                    "l2_leaf_reg": randint(1, 10)
                }
            }
             
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, 
                models=models, 
                param_distributions=param_distributions
                )

            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found.", sys)

            logging.info(
                f"Best model selected: {best_model_name} "
                f"with R2 score: {best_model_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)
            
            return model_r2_score
        
        except Exception as e:
            raise CustomException(e, sys)
