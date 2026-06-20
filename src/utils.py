import sys
import os
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param_distributions):
    try:
        report = {}

        for model_name, model in models.items():
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions[model_name],
                n_iter=10,
                cv=3,
                scoring='r2',
                random_state=42,
                n_jobs=1,
                verbose=1
            )
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_
            logging.info(
                            f"{model_name} Best Params: {random_search.best_params_}"
                        )



            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            models[model_name] = best_model
            
            logging.info(
                            f"{model_name} Train Score: {train_model_score:.4f}, "
                            f"Test Score: {test_model_score:.4f}"
                        )

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
