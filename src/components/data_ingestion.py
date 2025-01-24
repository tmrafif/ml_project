import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from data_transformation import DataTransformation
from model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function is responsible for ingesting the data from the given csv file
        and splitting it into train and test datasets. It also creates a raw_data csv
        which contains the entire dataset without any splitting.
        
        Returns:
            tuple: A tuple containing the paths of the train and test datasets.
        """
        logging.info('Data Ingestion methods Starts')
        try:
            logging.info('Read dataset as dataframe')
            df = pd.read_csv('notebook/data/stud.csv')

            os.makedirs(
                name=os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df,
                test_size=0.2,
                random_state=42
            )
            train_set.to_csv(self.ingestion_config.train_data_path,
                index=False,
                header=True
            )
            test_set.to_csv(self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info('Ingestion of the data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path    
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))