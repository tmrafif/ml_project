import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function returns a ColumnTransformer object which is used to
        transform the given dataset. The ColumnTransformer object is
        composed of two Pipelines, one for numerical features and one for
        categorical features. The numerical features are scaled using
        StandardScaler, while the categorical features are encoded using
        OneHotEncoder and then scaled using StandardScaler.

        Returns:
            ColumnTransformer: A ColumnTransformer object to transform the
            given dataset.
        """
        try:
            num_cols = [
                'writing_score',
                'reading_score'
            ]
            cat_cols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Numerical features standard scaling completed')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical features encoding completed')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            target_col_name = 'math_score'
            num_cols = [
                'writing_score',
                'reading_score'
            ]

            logging.info('Spliting feature and target columns')
            X_train = train_df.drop(columns=target_col_name, axis=1)
            y_train = train_df[target_col_name]

            X_test = test_df.drop(columns=target_col_name, axis=1)
            y_test = test_df[target_col_name]

            logging.info(
                'Applying preprocessing object on train and test dataframe'
            )
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            logging.info('Saved preprocessing object')
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


