import os
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from exception import CustomException

def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a specified file using dill serialization.

    Parameters:
    file_path (str): The file path where the object should be saved.
    obj (object): The Python object to be serialized and saved.

    Raises:
    CustomException: If an exception occurs during the file operation or serialization.
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Evaluates the performance of multiple regression models on training and test data using the R-squared metric.

    Parameters:
    X_train (array-like): The training data features.
    y_train (array-like): The training data target.
    X_test (array-like): The test data features.
    y_test (array-like): The test data target.
    models (dict): A dictionary of model names and their corresponding model objects.

    Returns:
    dict: A dictionary containing the R-squared scores for each model on the test data.

    Raises:
    CustomException: If an exception occurs during evaluation.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

    