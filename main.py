from PredictiveMaintenance.logging import logger
from PredictiveMaintenance.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from PredictiveMaintenance.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from PredictiveMaintenance.pipeline.stage_03_data_transformation_training import DataTransformationTrainingPipeline
from PredictiveMaintenance.pipeline.stage_04_training_validation_insertion import DataInsertionTrainingPipeline
from PredictiveMaintenance.pipeline.stage_05_data_preprocessing_training import DataPreProcessingTrainingPipeline
from PredictiveMaintenance.pipeline.stage_06_model_training import ModelTrainingPipeline


# STAGE_NAME = f"""Data Ingestion"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = f"""Data Validation Training"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_validation_training = DataValidationTrainingPipeline()
#     data_validation_training.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = f"""Data Transformation Training"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_transformation_training = DataTransformationTrainingPipeline()
#     data_transformation_training.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = f"""Data Insertion Training"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_insertion_training = DataInsertionTrainingPipeline()
#     data_insertion_training.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed, and data exported as csv for training <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = f"""Data Pre-Processing Training"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_preprocessing_training = DataPreProcessingTrainingPipeline()
#     data_preprocessing_training.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed, and data exported as csv for model building <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = f"""Model Training"""
try:
    logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed. <<<<<<<\n\n""")
except Exception as e:
    logger.exception(e)
    raise e
