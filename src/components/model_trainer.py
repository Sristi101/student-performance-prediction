import os
import sys
from dataclasses import dataclass


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

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


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
                "Random Forest": RandomForestRegressor(
                                                        max_depth=8,
                                                        max_features=None,
                                                        min_samples_leaf=3,
                                                        min_samples_split=2,
                                                        n_estimators=104,
                                                        random_state=42
                                                        ),
                "Decision Tree": DecisionTreeRegressor( 
                                                        max_depth=16,
                                                        max_features=None,
                                                        min_samples_leaf=8,
                                                        min_samples_split=16,
                                                        random_state=42
                                                        ),
                "Gradient Boosting": GradientBoostingRegressor(
                                                        learning_rate=0.012119891565915222,
                                                        max_depth=3,
                                                        min_samples_split=18,
                                                        n_estimators=574,
                                                        subsample=0.8447411578889518,
                                                        random_state=42
                                                        ),
                "XGB Regressor": XGBRegressor(
                                                        colsample_bytree=0.8832290311184181,
                                                        learning_rate=0.016175348288740735,
                                                        max_depth=4,
                                                        min_child_weight=8,
                                                        n_estimators=591,
                                                        subsample=0.9754210836063001,
                                                        objective='reg:squarederror',
                                                        random_state=42
                                                        ),
                "CatBoost Regressor": CatBoostRegressor(
                                                        depth=5,
                                                        iterations=753,
                                                        l2_leaf_reg=3,
                                                        learning_rate=0.019428755706020276,
                                                        verbose=False,
                                                        random_state=42
                                                        ),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models=models)

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
