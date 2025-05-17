import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

from src.constants import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    artifacts_folder: str = os.path.join(artifact_folder)
    transformed_train_file_path: str = os.path.join(artifact_folder, 'train.npy')
    transformed_test_file_path: str = os.path.join(artifact_folder, 'test.npy')
    transformed_object_file_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(feature_store_file_path)

            df.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        # sourcery skip: inline-immediately-returned-variable

        try:
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                    ('imputer', imputer_step),
                    ('scaler', scaler_step)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):

        logging.info("Entered initiate data transformation method of data transformation class")

        try:
            dataframe = self.get_data(feature_store_file_path = self.feature_store_file_path)

            x = dataframe.drop
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            preprocessor = self.get_data_transformer_object()
            x_train_scaled = preprocessor.fit_transform(x_train)
            x_test_scaled = preprocessor.transform(x_test)

            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            self.utils.save_path(file_path = preprocessor_path, obj=preprocessor)

            train_arr = np.c_[x_train_scaled, np.array(y_train)]
            test_arr = np.c_[x_test_scaled, np.array(y_test)]

            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys) from e