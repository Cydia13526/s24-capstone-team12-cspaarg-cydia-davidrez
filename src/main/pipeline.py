import os

import pandas as pd
import numpy as np
import statsmodels.api as sm
import xgboost as xgb
import pickle 
from typing import List
from sklearn.metrics import mean_absolute_error
import sqlite3
from database_setup import create_historical_database, create_forecast_database, create_predictions_database
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json

def load_data(processed_df_filepath: str):
    original_data = pd.read_csv(processed_df_filepath, parse_dates=['date'])
    original_data.set_index('date', inplace=True)
    
    data_diff = original_data.diff().dropna()
    # print(f'Shape of original data {original_data.shape}')
    # print(f'Date range of original data: {original_data.index.min()} to {original_data.index.max()}')
    # print(f'Shape of differenced data {data_diff.shape}')
    # print(f'Date range of differenced data: {data_diff.index.min()} to {data_diff.index.max()}')
    
    return original_data, data_diff


def find_best_chain(predictive_chain: List[str], data: pd.DataFrame, max_vars: int = None, pval_threshold: float = 0.10):
    results_df = pd.DataFrame(columns=['Variable', 'Adj-R-squared', 'Coefficient', 'P-value', 'Max VIF'])

    if max_vars is None:
        max_vars = len(data.columns) - len(predictive_chain)

    for _ in range(max_vars):
        temp_results_df = pd.DataFrame(columns=['Variable', 'Adj-R-squared', 'Coefficient', 'P-value', 'Max VIF'])
        added = False

        for column in data.columns:
            if column not in predictive_chain:
                X = data[predictive_chain]
                X = sm.add_constant(X)
                Y = data[column]

                model = sm.OLS(Y, X).fit()
                adj_rsquared = model.rsquared_adj
                coefficients = model.params
                pvalues = model.pvalues

                # Access p-value of the last predictor
                last_var = predictive_chain[-1]
                if pvalues[last_var] > pval_threshold:
                    continue

                vif_data = pd.DataFrame()
                vif_data['feature'] = X.columns
                vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

                max_vif = vif_data['VIF'].max()

                vif_threshold = 15.0
                if max_vif > vif_threshold:
                    continue

                current_result = pd.DataFrame({
                    'Variable': [column],
                    'Adj-R-squared': [adj_rsquared],
                    'Coefficient': [coefficients[last_var]],
                    'P-value': [pvalues[last_var]],
                    'Max VIF': [max_vif]
                })
                temp_results_df = pd.concat([temp_results_df, current_result], ignore_index=True)
                added = True

        if not added:
            print("No variables met the selection criteria.")
            break

        # Select the best variable based on Adjusted R-squared
        temp_results_df.sort_values(by='Adj-R-squared', inplace=True, ascending=False)
        best_var = temp_results_df.iloc[0]['Variable']
        predictive_chain.append(best_var)
        
        # Update results
        results_df = pd.concat([results_df, temp_results_df.loc[temp_results_df['Variable'] == best_var]], ignore_index=True)
    
    print("Final predictive chain:", predictive_chain)
    print("Model results:\n", results_df)

    return predictive_chain

def add_engineered_features(diff_df: pd.DataFrame, orig_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features based on FEDFUNDS to the differenced DataFrame.

    Parameters
    ----------
    diff_df : pd.DataFrame
        DataFrame containing differenced macroeconomic variables, including 'FEDFUNDS'.
    orig_df : pd.DataFrame
        Original DataFrame before differencing, used to reconstruct actual values.

    Returns
    -------
    diff_df : pd.DataFrame
        DataFrame with additional engineered features.
    """

    # Ensure 'FEDFUNDS' is in diff_df
    if 'FEDFUNDS' not in diff_df.columns:
        raise ValueError("FEDFUNDS column not found in diff_df")

    # 1. Reconstruct FEDFUNDS actual values from differenced data
    first_fedfunds = orig_df['FEDFUNDS'].iloc[0] if 'FEDFUNDS' in orig_df.columns else 0
    diff_df['FEDFUNDS_Actual'] = diff_df['FEDFUNDS'].cumsum() + first_fedfunds

    # 2. Calculate FEDFUNDS percentage change
    diff_df['FEDFUNDS_pct_change'] = diff_df['FEDFUNDS_Actual'].replace(0, np.nan).pct_change() * 100

    # 3. Calculate duration of constant FEDFUNDS values
    # Optimized calculation using pandas
    diff_df['FEDFUNDS_Duration'] = (diff_df['FEDFUNDS_Actual']
                                    .ne(diff_df['FEDFUNDS_Actual'].shift())
                                    .cumsum()
                                    .groupby(diff_df['FEDFUNDS_Actual'].ne(diff_df['FEDFUNDS_Actual'].shift()).cumsum())
                                    .cumcount() + 1)

    # 4. Compute Exponentially Weighted Moving Average
    diff_df['FEDFUNDS_Weighted'] = diff_df['FEDFUNDS_Actual'].ewm(span=12, adjust=False).mean()

    # 5. Calculate Time Since Last Change in FEDFUNDS
    # When FEDFUNDS differenced value is non-zero, a change occurred
    diff_df['Time_Since_Change'] = diff_df['FEDFUNDS'].eq(0).cumsum() - diff_df['FEDFUNDS'].eq(0).cumsum().where(diff_df['FEDFUNDS'].ne(0)).ffill().fillna(0).astype(int)

    # Alternatively, use a custom function
    # def time_since_last_change(series):
    #     time_since_change = []
    #     count = 0
    #     for change in (series != 0):
    #         if change:
    #             count = 0
    #         else:
    #             count += 1
    #         time_since_change.append(count)
    #     return pd.Series(time_since_change, index=series.index)
    # diff_df['Time_Since_Change'] = time_since_last_change(diff_df['FEDFUNDS'])

    # 6. Create interaction term between FEDFUNDS_Actual and FEDFUNDS_Duration
    diff_df['FEDFUNDS_interaction'] = diff_df['FEDFUNDS_Actual'] * diff_df['FEDFUNDS_Duration']

    # 7. Calculate FEDFUNDS Deviation from Expanding Mean
    diff_df['FEDFUNDS_Expanding_Mean'] = diff_df['FEDFUNDS_Actual'].expanding().mean()
    diff_df['FEDFUNDS_Deviation'] = diff_df['FEDFUNDS_Actual'] - diff_df['FEDFUNDS_Expanding_Mean']

    # 8. Compute Rolling Mean and Standard Deviation
    diff_df['FEDFUNDS_Rolling_mean'] = diff_df['FEDFUNDS_Actual'].rolling(window=12, min_periods=1).mean()
    diff_df['FEDFUNDS_Rolling_std'] = diff_df['FEDFUNDS_Actual'].rolling(window=12, min_periods=1).std()

    # 9. Create Lag Features for FEDFUNDS and FEDFUNDS_Actual
    for lag in range(1, 13):  # 12 periods lag
        diff_df[f'FEDFUNDS_lag_{lag}'] = diff_df['FEDFUNDS'].shift(lag)
        # diff_df[f'FEDFUNDS_Actual_lag_{lag}'] = diff_df['FEDFUNDS_Actual'].shift(lag)

    # 10. Additional Engineered Features
    # Interest Rate Change Direction
    diff_df['FEDFUNDS_Direction'] = diff_df['FEDFUNDS'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Absolute Change in FEDFUNDS
    diff_df['FEDFUNDS_Absolute_Change'] = diff_df['FEDFUNDS'].abs()

    # Rolling Volatility of FEDFUNDS
    diff_df['FEDFUNDS_Rolling_Volatility'] = diff_df['FEDFUNDS'].rolling(window=12, min_periods=1).std()

    # # Yield Spread (if 'DGS10' and 'DGS2' are available in diff_df)
    # if 'DGS10' in diff_df.columns and 'DGS2' in diff_df.columns:
    #     diff_df['Yield_Spread'] = diff_df['DGS10'] - diff_df['DGS2']

    if 'UNRATE' in diff_df.columns:
        diff_df['FEDFUNDS_UNRATE_interaction'] = diff_df['FEDFUNDS'] * diff_df['UNRATE']
        
    # 11. Handle Missing Values
    # Fill NaNs selectively
    diff_df['FEDFUNDS_pct_change'].fillna(0, inplace=True)
    diff_df['FEDFUNDS_Duration'].fillna(0, inplace=True)
    diff_df['Time_Since_Change'].fillna(0, inplace=True)
    diff_df['FEDFUNDS_interaction'].fillna(0, inplace=True)
    diff_df['FEDFUNDS_Deviation'].fillna(0, inplace=True)
    diff_df['FEDFUNDS_Direction'].fillna(0, inplace=True)
    diff_df['FEDFUNDS_Absolute_Change'].fillna(0, inplace=True)
    diff_df['FEDFUNDS_Rolling_Volatility'].fillna(0, inplace=True)

    # For lag features, initial NaNs are expected; decide whether to fill or drop
    # Here, we'll drop rows with NaNs resulting from lagging
    diff_df.dropna(inplace=True)
    return diff_df
    

def train_model(best_chain, original_data):
    feature_importances = {}
    for variable in best_chain[1:]:  # Excluding FEDFUNDS
        print(f"Training model for {variable}")
        
        X = original_data[best_chain[:best_chain.index(variable)]]
        y = original_data[variable]

        # Calculate differenced data
        X_diff = X.diff().dropna()
        y_diff = y.diff().dropna()

        # Add engineered features
        X_diff = add_engineered_features(X_diff, original_data)
        
        # **Realign y_diff with X_diff**
        y_diff = y_diff.loc[X_diff.index]

        train_size = int(len(X_diff) * 0.8)
        X_train, X_test = X_diff[:train_size], X_diff[train_size:]
        y_train, y_test = y_diff[:train_size], y_diff[train_size:]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            max_depth=5,
            n_estimators=1000,)
        
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"X_train index range: {X_train.index.min()} to {X_train.index.max()}")
        print(f"X_test index range: {X_test.index.min()} to {X_test.index.max()}")
        print(f"features: {X_train.columns}")
      
        predictions = []
        for i in range(len(X_test)):
            model.fit(X_train, y_train)
            pred = model.predict(X_test.iloc[i:i+1]) #1 row prediction
            predictions.append(pred[0])
            
            # Add the actual test value to the training set for the next iteration
            X_train = pd.concat([X_train, X_test.iloc[i:i+1]])
            y_train = pd.concat([y_train, pd.Series(y_test.iloc[i], index=[X_test.index[i]])])
        
        mae_test = mean_absolute_error(y_test, predictions)
        print(f"Mean Absolute Error for {variable}: {mae_test}")

        # Get the last actual value before predictions start
        last_actual_value = original_data[variable].iloc[train_size]

        # Convert predictions from differences back to real values
        real_predictions = {}
        cumulative_sum = last_actual_value
        for date, diff_value in zip(X_test.index, predictions):
            cumulative_sum += diff_value
            real_predictions[date] = cumulative_sum
            # print(f"Date: {date}, Prediction: {cumulative_sum}")

        # Insert predictions for this variable into the database
        insert_predictions(variable, real_predictions)

        os.makedirs(os.path.join(os.getcwd(), "src/resources/models"))
        model_filename = f"src/resources/models/best_model_{variable}.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved as {model_filename}")

        # After training the model, get feature importances
        importances = dict(zip(X_train.columns, model.feature_importances_))
        # print(f"Raw importances for {variable}:")
        # print(importances)
        
        # Convert float32 to regular float
        importances = {k: float(v) for k, v in importances.items()}
        
        # Sort and get top 10 features
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        feature_importances[variable] = dict(sorted_importances)
        
        # print(f"Top 10 importances for {variable}:")
        # print(feature_importances[variable])

    # Save feature importances to a JSON file
    with open('resources/models/feature_importances.json', 'w') as f:
        json.dump(feature_importances, f, indent=4)

    return

def insert_predictions(variable, predictions):  
    conn = sqlite3.connect(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db"))
    cursor = conn.cursor()

    # Ensure the column exists
    cursor.execute(f"PRAGMA table_info(predictions_macro)")
    columns = [row[1] for row in cursor.fetchall()]
    if f"{variable}_prediction" not in columns:
        cursor.execute(f"ALTER TABLE predictions_macro ADD COLUMN {variable}_prediction REAL")

    # Insert or update predictions
    for date, value in predictions.items():
        cursor.execute(f"""
        INSERT INTO predictions_macro (date, {variable}_prediction)
        VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET
        {variable}_prediction = excluded.{variable}_prediction
        """, (date.strftime('%Y-%m-%d'), value))

    conn.commit()
    conn.close()
    print(f"Predictions for {variable} inserted into the database.")

def insert_historical_data_to_db(data: pd.DataFrame):
    # Ensure the database exists
    create_historical_database()

    # Connect to the database
    conn = sqlite3.connect(os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db"))
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS historical_macro
    (date TEXT PRIMARY KEY)
    ''')

    # Add columns for each macro if they don't exist
    for column in data.columns:
        cursor.execute(f"PRAGMA table_info(historical_macro)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE historical_macro ADD COLUMN {column} REAL")

    # Prepare data for insertion
    data_to_insert = data.reset_index().to_dict(orient='records')

    # Insert data into the database
    for row in data_to_insert:
        placeholders = ', '.join(['?' for _ in row])
        columns = ', '.join(row.keys())
        sql = f"INSERT OR REPLACE INTO historical_macro ({columns}) VALUES ({placeholders})"
        
        # Convert Timestamp to string
        values = [str(value) if isinstance(value, pd.Timestamp) else value for value in row.values()]
        
        cursor.execute(sql, values)

    conn.commit()
    conn.close()
    print("Historical data inserted into the database.")

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
    insert_historical_data_to_db(original_data)

    print("------- Step Up Pipeline - Step 3: Find Best Chain -------")
    initial_chain = ['FEDFUNDS']
    best_chain = find_best_chain(initial_chain, differenced_data)
    print("Best chain:", best_chain)

    print("------- Step Up Pipeline - Step 4: Start Training the Model -------")
    train_model(best_chain, original_data)

    return original_data, best_chain


if __name__ == '__main__':
    start_pipeline()




