import os, sys
import pandas as pd
import sqlite3
import streamlit as st
sys.path.append(os.path.join(os.getcwd(), "src"))
from streamlit_app.configs.logger_config import logger

class MongoDataLoader:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def _execute_query(self, query, parse_dates=None, error_message=None):
        try:
            df = pd.read_sql_query(query, self.conn, parse_dates=parse_dates)
            return df
        except sqlite3.OperationalError as e:
            logger.info(f"SQLite error: {str(e)}")
            if error_message:
                st.warning(f"{error_message}: {str(e)}")
        except Exception as e:
            logger.info(f"Unexpected error: {str(e)}")
            if error_message:
                st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame()

    def _process_data(self, df, date_column):
        if not df.empty:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
        return df

    def load_historical_data(self, variables):
        if variables:
            query = f"SELECT date, {', '.join(variables)} FROM historical_macro"
        else:
            query = "SELECT * FROM historical_macro"
        df = self._execute_query(query)
        return self._process_data(df, 'date')

    def load_prediction_data(self, variables):
        if not variables:
            raise ValueError("No variables specified for prediction data.")

        variable = variables[0]
        query = f"SELECT date, {variable}_prediction FROM predictions_macro"
        df = self._execute_query(query, parse_dates=['date'], error_message=f"No predictions available for {variable}")
        return self._process_data(df, 'date')

    def insert_historical_data(self, data: pd.DataFrame):
        cursor = self.conn.cursor()

        for column in data.columns:
            cursor.execute(f"PRAGMA table_info(historical_macro)")
            existing_columns = [row[1] for row in cursor.fetchall()]
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE historical_macro ADD COLUMN {column} REAL")

        data_to_insert = data.reset_index().to_dict(orient='records')

        for row in data_to_insert:
            placeholders = ', '.join(['?' for _ in row])
            columns = ', '.join(row.keys())
            sql = f"INSERT OR REPLACE INTO historical_macro ({columns}) VALUES ({placeholders})"

            values = [str(value) if isinstance(value, pd.Timestamp) else value for value in row.values()]

            cursor.execute(sql, values)

        self.conn.commit()
        logger.info("Historical data inserted into the database.")

    def insert_predictions_data(self, variable, predictions):
        cursor = self.conn.cursor()

        cursor.execute(f"PRAGMA table_info(predictions_macro)")
        columns = [row[1] for row in cursor.fetchall()]
        if f"{variable}_prediction" not in columns:
            cursor.execute(f"ALTER TABLE predictions_macro ADD COLUMN {variable}_prediction REAL")

        for date, value in predictions.items():
            cursor.execute(f"""
            INSERT INTO predictions_macro (date, {variable}_prediction)
            VALUES (?, ?)
            ON CONFLICT(date) DO UPDATE SET
            {variable}_prediction = excluded.{variable}_prediction
            """, (date.strftime('%Y-%m-%d'), value))

        self.conn.commit()
        logger.info(f"Predictions for {variable} inserted into the database.")

    def insert_forecast_data(self, variable, forecast_data):
        cursor = self.conn.cursor()

        cursor.execute("PRAGMA table_info(forecasts_macro)")
        columns = [row[1] for row in cursor.fetchall()]
        if f"{variable}_forecast" not in columns:
            cursor.execute(f"ALTER TABLE forecasts_macro ADD COLUMN {variable}_forecast REAL")

        for date, value in forecast_data.items():
            cursor.execute(f"""
            INSERT INTO forecasts_macro (date,  model, {variable}_forecast)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
            {variable}_forecast = excluded.{variable}_forecast
            """, (date.strftime('%Y-%m-%d'),  'user_forecast',  float(value)))

        self.conn.commit()
        logger.info(f"Forecasts for {variable} stored in the database.")
