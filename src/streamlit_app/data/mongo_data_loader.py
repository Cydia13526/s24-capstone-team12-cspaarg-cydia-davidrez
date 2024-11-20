import os, sys
import pandas as pd
import sqlite3
import streamlit as st
sys.path.append(os.path.join(os.getcwd(), "src"))
from streamlit_app.configs.logger_config import logger

class MongoDataLoader:
    """
    A class for loading, processing, and inserting macroeconomic data into an SQLite database.

    This class provides methods to load historical data, prediction data, and forecast data
    from an SQLite database and insert them back into the respective tables. It also supports
    handling errors and logging for better traceability.

    Attributes:
        db_path (str): The file path to the SQLite database.
        conn (sqlite3.Connection): The SQLite database connection.

    Methods:
        __enter__(): Initializes the connection to the SQLite database.
        __exit__(exc_type, exc_val, exc_tb): Closes the database connection when exiting the context.
        _execute_query(query, parse_dates=None, error_message=None): Executes an SQL query and returns a DataFrame.
        _process_data(df, date_column): Processes the data by converting the date column to datetime and setting it as the index.
        load_historical_data(variables): Loads historical macroeconomic data from the database.
        load_prediction_data(variables): Loads prediction data from the database for a specified variable.
        insert_historical_data(data): Inserts or updates historical data in the database.
        insert_predictions_data(variable, predictions): Inserts or updates predictions data for a specified variable.
        insert_forecast_data(variable, forecast_data): Inserts or updates forecast data for a specified variable.
    """
    def __init__(self, db_path):
        """
         Initializes the MongoDataLoader instance with the database path.

         Args:
             db_path (str): The path to the SQLite database file.
         """
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """
        Establishes a connection to the SQLite database when entering the context.

        Returns:
          MongoDataLoader: The instance of MongoDataLoader with an open database connection.
        """
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the database connection when exiting the context.

        Args:
            exc_type (type): The exception type if an exception occurred.
            exc_val (Exception): The exception instance if an exception occurred.
            exc_tb (traceback): The traceback of the exception if one occurred.
        """
        if self.conn:
            self.conn.close()

    def _execute_query(self, query, parse_dates=None, error_message=None):
        """
        Executes a SQL query and returns the results as a Pandas DataFrame.

        Args:
            query (str): The SQL query to execute.
            parse_dates (list, optional): List of columns to parse as dates. Defaults to None.
            error_message (str, optional): A custom error message to display in case of an error. Defaults to None.

        Returns:
            pd.DataFrame: The resulting DataFrame from the query.
        """
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
        """
        Processes the DataFrame by converting the date column to datetime and setting it as the index.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            date_column (str): The name of the column containing date values.

        Returns:
            pd.DataFrame: The processed DataFrame with the date column as the index.
        """
        if not df.empty:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
        return df

    def load_historical_data(self, variables):
        """
        Loads historical macroeconomic data from the database.

        Args:
            variables (list): A list of variables to retrieve from the database.

        Returns:
            pd.DataFrame: The historical data for the specified variables.
        """
        if variables:
            query = f"SELECT date, {', '.join(variables)} FROM historical_macro"
        else:
            query = "SELECT * FROM historical_macro"
        df = self._execute_query(query)
        return self._process_data(df, 'date')

    def load_prediction_data(self, variables):
        """
        Loads prediction data for a specified variable from the database.

        Args:
            variables (list): A list containing the variable to load predictions for.

        Returns:
            pd.DataFrame: The prediction data for the specified variable.

        Raises:
            ValueError: If no variables are specified.
        """
        if not variables:
            raise ValueError("No variables specified for prediction data.")

        variable = variables[0]
        query = f"SELECT date, {variable}_prediction FROM predictions_macro"
        df = self._execute_query(query, parse_dates=['date'], error_message=f"No predictions available for {variable}")
        return self._process_data(df, 'date')

    def insert_historical_data(self, data: pd.DataFrame):
        """
        Inserts or updates historical macroeconomic data into the database.

        Args:
            data (pd.DataFrame): The historical data to insert or update.
        """
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
        """
        Inserts or updates prediction data for a specified variable into the database.

        Args:
            variable (str): The variable for which predictions are being inserted.
            predictions (dict): A dictionary where the keys are dates and values are predicted values.
        """
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
        """
        Inserts or updates forecast data for a specified variable into the database.

        Args:
            variable (str): The variable for which forecast data is being inserted.
            forecast_data (dict): A dictionary where the keys are dates and values are forecasted values.
        """
        cursor = self.conn.cursor()
        self.create_forecast_table()

        cursor.execute("PRAGMA table_info(forecast_macro)")
        columns = [row[1] for row in cursor.fetchall()]

        if f"{variable}_forecast" not in columns:
            cursor.execute(f"ALTER TABLE forecast_macro ADD COLUMN {variable}_forecast REAL")

        forecast_data_dict = dict(forecast_data)

        for date, value in forecast_data_dict.items():
            cursor.execute(f"""
            INSERT INTO forecast_macro (date, model, {variable}_forecast)
            VALUES (?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
            {variable}_forecast = excluded.{variable}_forecast
            """, (date.strftime('%Y-%m-%d'), 'user_forecast', float(value)))

        self.conn.commit()
        logger.info(f"Forecasts for {variable} stored in the database.")