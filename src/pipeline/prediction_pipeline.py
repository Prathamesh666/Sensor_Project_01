import shutil
import sys
import os
import pandas as pd
import pickle
import src.logger as logging
import src.exception as CustomException
from flask import request
from src.constants import *
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dir_name: str = "predictions"
    prediction_file_name: str = "predictions_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dir_name, prediction_file_name)

class PredictionPipeline:
    def __init__(self, request: request): # type: ignore
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()
        
    def save_input_files(self) -> str:
        try:
            pred_file_input_dir = "prediction artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path)

            return pred_file_path
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def predict(self, features):
        try:
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(file_path=self.prediction_pipeline_config.preprocessor_path)

            transformed_x = preprocessor.transform(features)

            return model.predict(transformed_x)
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def get_predicted_dataframe(self, input_dataframe_path: pd.DataFrame):
        try:
            prediction_column_name: str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)

            input_dataframe.drop(columns="unnamed: 0") if "unnamed: 0" in input_dataframe.columns else None

            predictions = self.predict(input_dataframe)

            input_dataframe[prediction_column_name] = list(predictions)

            target_column_mapping = {0:'bad', 1:'good'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)

            input_dataframe.to.csv(self.prediction_pipeline_config.prediction_file_path, index=False)

            logging.info(f"Predictions Completed")

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys) from e  