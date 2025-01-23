import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
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
    obj.initiate_data_ingestion()