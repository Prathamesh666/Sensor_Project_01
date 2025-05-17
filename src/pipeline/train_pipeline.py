import sys
import os

from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


class TrainingPipeline:

    def start_data_ingestion(self):
        # sourcery skip: inline-immediately-returned-variable
        try:
            data_ingestion = DataIngestion()
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_data_transformation(self, feature_store_file_path):
        
        try:
            data_transformation = DataTransformation(feature_store_file_path = feature_store_file_path)
            train_array, test_array, preprocessor = data_transformation.initiate_data_transformation()
            return train_array, test_array, preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def start_model_training(self, train_array, test_array):
        # sourcery skip: inline-immediately-returned-variable
        try:
            model_trainer = ModelTrainer()
            return model_trainer.initiate_model_trainer(train_array, test_array)
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_array, test_array, preprocessor = self.start_data_transformation(feature_store_file_path)
            r2_square = self.start_model_training(train_array, test_array)

            print ("training completed. Trained model score is : ", r2_square)
            
        except Exception as e:
            raise CustomException(e, sys) from e