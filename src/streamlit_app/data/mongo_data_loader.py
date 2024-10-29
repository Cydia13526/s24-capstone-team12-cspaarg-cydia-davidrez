import os
import pandas as pd
import sqlite3
import streamlit as st

class MongoDataLoader:
    def __init__(self, db_path):
        """
        Initializes the MongoDataLoader with the given variables and database path.

        :param variables: List of variables to load from the database.
        :param db_path: Path to the SQLite database.
        """
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """Establishes a database connection when entering the context."""
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the database connection when exiting the context."""
        if self.conn:
            self.conn.close()

    def _execute_query(self, query, parse_dates=None, error_message=None):
        """
        Executes an SQL query and handles errors.

        :param query: The SQL query to execute.
        :param parse_dates: Columns to parse as dates (optional).
        :param error_message: Custom error message for exceptions (optional).
        :return: A pandas DataFrame with the query results.
        """
        try:
            df = pd.read_sql_query(query, self.conn, parse_dates=parse_dates)
            return df
        except sqlite3.OperationalError as e:
            print(f"SQLite error: {str(e)}")
            if error_message:
                st.warning(f"{error_message}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if error_message:
                st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame()

    def _process_data(self, df, date_column):
        """
        Processes the loaded data: converts date column to datetime, sets the index, and sorts the index.

        :param df: The DataFrame to process.
        :param date_column: The column name to convert to datetime and set as index.
        :return: A processed pandas DataFrame.
        """
        if not df.empty:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df.sort_index(inplace=True)
        return df

    def load_historical_data(self, variables):
        """
        Loads historical macroeconomic data from the database.

        :return: A pandas DataFrame with historical data, indexed by date.
        """
        if variables:
            query = f"SELECT date, {', '.join(variables)} FROM historical_macro"
        else:
            query = "SELECT * FROM historical_macro"
        df = self._execute_query(query)
        return self._process_data(df, 'date')

    def load_prediction_data(self, variables):
        """
        Loads prediction data for the specified variable from the database.

        :return: A pandas DataFrame with prediction data, indexed by date.
        """
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
        print("Historical data inserted into the database.")

    def insert_predictions(self, variable, predictions):
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
        print(f"Predictions for {variable} inserted into the database.")
