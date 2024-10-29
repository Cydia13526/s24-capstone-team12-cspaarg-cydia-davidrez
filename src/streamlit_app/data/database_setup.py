import sqlite3
import os, sys
sys.path.append(os.path.join(os.getcwd(), "src"))
from streamlit_app.data.fred_data_loader import FREDDataLoader

def delete_database(db_name):
    if os.path.exists(f'resources/data/mongo_db/{db_name}.db'):
        os.remove(f'resources/data/mongo_db/{db_name}.db')
        print(f"{db_name} database deleted.")

def create_database(db_path, create_table_query):
    delete_database(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(create_table_query)
    return conn

def create_historical_database(conn = None):
    try:
        db_path = os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")

        loader = FREDDataLoader()
        df = loader.load_data()

        preprocessed_df = loader.preprocess_data(df)
        variables = preprocessed_df.columns

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS historical_macro (
            date TEXT PRIMARY KEY,
            {}
        )
        '''.format(',\n        '.join([f'{var} REAL' for var in variables]))

        conn = create_database(db_path, create_table_query)
        preprocessed_df.to_sql('historical_macro', conn, if_exists='replace', index=True, index_label='date')
        print("Historical database created and populated.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.commit()
        conn.close()

def create_forecast_database(conn = None):
    try:
        db_path = os.path.join(os.getcwd(), "src/resources/data/mongo_db/forecast_macro.db")

        loader = FREDDataLoader()
        variables = loader.variables

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS forecast_macro (
            date TEXT,
            model TEXT,
            {}
        )
        '''.format(',\n        '.join([f'{var}_forecast REAL' for var in variables]))

        conn = create_database(db_path, create_table_query)
        print("Forecast database created.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.commit()
        conn.close()

def create_predictions_database(conn = None):
    try:
        db_path = os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")

        loader = FREDDataLoader()
        variables = loader.variables

        create_table_query = '''
        CREATE TABLE IF NOT EXISTS predictions_macro (
            date TEXT PRIMARY KEY,
            {}
        )
        '''.format(',\n        '.join([f'{var}_prediction REAL' for var in variables]))
        conn = create_database(db_path, create_table_query)
        print("Predictions database created.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.commit()
        conn.close()

if __name__ == "__main__":
    create_historical_database()
    create_forecast_database()
    create_predictions_database()