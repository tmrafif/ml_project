import os
import sys

import dill
import numpy as np
import pandas as pd

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