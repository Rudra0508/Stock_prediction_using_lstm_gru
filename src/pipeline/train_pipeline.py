import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("🚀 Training Pipeline Started")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            raw_data_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed: {raw_data_path}")

            # Step 2: Data Transformation
            transformation = DataTransformation()
            processed_data_path = transformation.initiate_data_transformation()
            logging.info(f"Data Transformation Completed: {processed_data_path}")

            # Step 3: Model Training
            trainer = ModelTrainer()
            trainer.initiate_model_training()
            logging.info("Model Training Completed")

            logging.info("✅ Training Pipeline Completed Successfully")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()