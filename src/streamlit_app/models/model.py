import json
import sys, os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.mongo_data_loader import MongoDataLoader


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
        with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")) as loader:
            loader.insert_predictions(variable, real_predictions)

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
