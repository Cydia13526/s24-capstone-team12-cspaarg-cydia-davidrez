import sqlite3
import os, sys
from contextlib import closing
sys.path.append(os.path.join(os.getcwd(), "src"))
from streamlit_app.data.fred_data_loader import FREDDataLoader
from streamlit_app.configs.logger_config import logger

def delete_database(db_name):
    db_path = f'resources/data/mongo_db/{db_name}.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"{db_name} database deleted.")

def create_and_populate_database(db_name, create_table_query, data=None):
    db_path = f'src/resources/data/mongo_db/{db_name}.db'
    delete_database(db_name)

    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        if data is not None:
            data.to_sql(db_name, conn, if_exists='replace', index=True, index_label='date')
        logger.info(f"{db_name} database created and populated.")

def create_historical_database():
    loader = FREDDataLoader()
    df = loader.load_data()
    preprocessed_df = loader.preprocess_data(df)
    variables = preprocessed_df.columns

    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS historical_macro (
        date TEXT PRIMARY KEY,
        {', '.join([f"{var} REAL" for var in variables])}
    )
    '''
    create_and_populate_database("historical_macro", create_table_query, data=preprocessed_df)

def create_forecast_database():
    loader = FREDDataLoader()
    variables = loader.variables

    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS forecast_macro (
        date TEXT,
        model TEXT,
        {', '.join([f"{var}_forecast REAL" for var in variables])}
    )
    '''
    create_and_populate_database("forecast_macro", create_table_query)

def create_predictions_database():
    loader = FREDDataLoader()
    variables = loader.variables

    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS predictions_macro (
        date TEXT PRIMARY KEY,
        {', '.join([f"{var}_prediction REAL" for var in variables])}
    )
    '''
    create_and_populate_database("predictions_macro", create_table_query)

if __name__ == "__main__":
    create_historical_database()
    create_forecast_database()
    create_predictions_database()