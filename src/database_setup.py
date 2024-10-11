import sqlite3
import pandas as pd
import os
from data_load import FREDDataLoader

def delete_database(db_name):
    if os.path.exists(f'data/{db_name}.db'):
        os.remove(f'data/{db_name}.db')
        print(f"{db_name} database deleted.")

def create_historical_database():
    delete_database('historical_macro')
    
    # Get data from FREDDataLoader
    loader = FREDDataLoader()
    df = loader.load_data()
    
    # Preprocess and resample the data
    preprocessed_df = loader.preprocess_data(df)
    
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('data/historical_macro.db')
    cursor = conn.cursor()

    variables = preprocessed_df.columns

    # Create table for historical data dynamically
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS historical_macro (
        date TEXT PRIMARY KEY,
        {}
    )
    '''.format(',\n        '.join([f'{var} REAL' for var in variables]))

    cursor.execute(create_table_query)

    # Insert data into the database
    preprocessed_df.to_sql('historical_macro', conn, if_exists='replace', index=True, index_label='date')

    conn.commit()
    conn.close()

    print("Historical database created and populated.")

def create_forecast_database():
    delete_database('forecast_macro')
    
    conn = sqlite3.connect('data/forecast_macro.db')
    cursor = conn.cursor()

    loader = FREDDataLoader()
    variables = loader.variables

    create_table_query = '''
    CREATE TABLE IF NOT EXISTS forecast_macro (
        date TEXT,
        model TEXT,
        {}
    )
    '''.format(',\n        '.join([f'{var}_forecast REAL' for var in variables]))

    cursor.execute(create_table_query)

    conn.commit()
    conn.close()

    print("Forecast database created.")

def create_predictions_database():
    delete_database('predictions_macro')
    
    conn = sqlite3.connect('data/predictions_macro.db')
    cursor = conn.cursor()

    loader = FREDDataLoader()
    variables = loader.variables

    create_table_query = '''
    CREATE TABLE IF NOT EXISTS predictions_macro (
        date TEXT PRIMARY KEY,
        {}
    )
    '''.format(',\n        '.join([f'{var}_prediction REAL' for var in variables]))

    cursor.execute(create_table_query)

    conn.commit()
    conn.close()

    print("Predictions database created.")

if __name__ == "__main__":
    create_historical_database()
    create_forecast_database()
    create_predictions_database()