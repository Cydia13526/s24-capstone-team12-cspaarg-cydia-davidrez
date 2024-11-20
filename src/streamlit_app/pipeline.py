import os
import sys
import pandas as pd
import warnings
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.components.chain import find_best_chain
from streamlit_app.models.model import train_model
from streamlit_app.utils.common_util import load_data
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.data.database_setup import create_historical_database, create_forecast_database, create_predictions_database
from streamlit_app.configs.logger_config import logger
warnings.filterwarnings("ignore")

def start_pipeline():
    """
      Executes the full data processing pipeline, including database setup, data loading, chain selection,
      and model training. The pipeline follows a series of steps to preprocess and analyze economic data,
      then trains a model based on the best chain identified.

      The pipeline consists of the following steps:
      1. **Database Setup**: Initializes databases for historical data, forecast data, and predictions.
      2. **Data Insertion**: Loads preprocessed economic data and inserts it into a MongoDB database.
      3. **Chain Selection**: Identifies the best chain from a predefined initial chain using the differenced data.
      4. **Model Training**: Trains a model using the best chain and the original dataset.

      Logs each step of the pipeline for tracking and debugging.

      Returns:
          tuple: A tuple containing:
              - original_data (pd.DataFrame): The original economic data loaded from the CSV file.
              - best_chain (list): The best chain identified for the model training.

      Side Effects:
          - Creates and configures databases using the `create_historical_database`, `create_forecast_database`,
            and `create_predictions_database` functions.
          - Inserts data into the MongoDB database using the `MongoDataLoader`.
          - Logs each step of the pipeline for progress tracking.
          - The model is trained based on the best chain selected.

      Example:
          original_data, best_chain = start_pipeline()

    """
    logger.info("------- Step Up Pipeline - Step 1: Database Set Up -------")
    create_historical_database()
    create_forecast_database()
    create_predictions_database()

    logger.info("------- Step Up Pipeline - Step 2: Insert Preprocessed Data into Database -------")
    original_data, differenced_data = load_data(os.path.join(os.getcwd(), "src/resources/data/processed/preprocessed_economic_data.csv"))
    original_data.index = pd.to_datetime(original_data.index)
    with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")) as loader:
        loader.insert_historical_data(original_data)

    logger.info("------- Step Up Pipeline - Step 3: Find Best Chain -------")
    initial_chain = ['FEDFUNDS']
    best_chain = find_best_chain(initial_chain, differenced_data)
    logger.info(f"Best chain: {best_chain}")

    logger.info("------- Step Up Pipeline - Step 4: Start Training the Model -------")
    train_model(best_chain, original_data)

    return original_data, best_chain

if __name__ == '__main__':
    start_pipeline()




