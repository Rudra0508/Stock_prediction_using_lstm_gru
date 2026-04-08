import sys
import os
import shutil

from utils.logger import logging
from utils.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:

    def __init__(self):
        pass

    # ---------------- CLEAN OLD ARTIFACTS ---------------- #
    def clean_artifacts(self):
        try:
            if os.path.exists("artifacts"):
                shutil.rmtree("artifacts")
                logging.info("🧹 Old artifacts removed successfully")
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- MAIN PIPELINE ---------------- #
    def run_pipeline(self):
        try:
            logging.info("🚀 Training Pipeline Started")

            # STEP 0: CLEAN
            self.clean_artifacts()

            # STEP 1: DATA INGESTION
            logging.info("📥 Step 1: Data Ingestion Started")
            ingestion = DataIngestion()
            raw_data_path = ingestion.initiate_data_ingestion()
            logging.info(f"✅ Data Ingestion Completed → {raw_data_path}")

            # STEP 2: DATA TRANSFORMATION
            logging.info("⚙️ Step 2: Data Transformation Started")
            transformation = DataTransformation()
            processed_data_path = transformation.initiate_data_transformation(raw_data_path)
            logging.info(f"✅ Data Transformation Completed → {processed_data_path}")

            # STEP 3: MODEL TRAINING
            logging.info("🧠 Step 3: Model Training Started")
            trainer = ModelTrainer()
            trainer.initiate_model_training(processed_data_path)
            logging.info("✅ Model Training Completed")

            logging.info("🎯 Training Pipeline Completed Successfully")

        except Exception as e:
            logging.error("❌ Error occurred in training pipeline")
            raise CustomException(e, sys)


# ---------------- RUN PIPELINE ---------------- #
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()