import os
import sys
import pandas as pd
import warnings
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.services.find_best_chain import find_best_chain
from streamlit_app.models.model import train_model
from streamlit_app.utils.common_util import load_data
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.data.database_setup import create_historical_database, create_forecast_database, create_predictions_database
warnings.filterwarnings("ignore")

def db_set_up():
    create_historical_database()
    create_forecast_database()
    create_predictions_database()

def start_pipeline():
    print("------- Step Up Pipeline - Step 1: Database Set Up -------")
    db_set_up()

    print("------- Step Up Pipeline - Step 2: Insert Preprocessed Data into Database -------")
    original_data, differenced_data = load_data(os.path.join(os.getcwd(), "src/resources/data/processed/preprocessed_economic_data.csv"))
    original_data.index = pd.to_datetime(original_data.index)
    with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")) as loader:
        loader.insert_historical_data(original_data)

    print("------- Step Up Pipeline - Step 3: Find Best Chain -------")
    initial_chain = ['FEDFUNDS']
    best_chain = find_best_chain(initial_chain, differenced_data)
    print("Best chain:", best_chain)

    print("------- Step Up Pipeline - Step 4: Start Training the Model -------")
    train_model(best_chain, original_data)

    return original_data, best_chain

if __name__ == '__main__':
    start_pipeline()




