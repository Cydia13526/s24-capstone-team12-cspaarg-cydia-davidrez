"""
This module trains machine learning models to forecast macroeconomic variables using
a time-series dataset and XGBoost regression. It performs data preparation, model training,
feature engineering, prediction, and result storage.

Usage:
1. Provide the best causal chain of variables and the original dataset to `train_model`.
2. Models are trained for each variable in the chain, storing predictions, models,
   and feature importances.

Functions:
- `prepare_data`: Prepares training and testing datasets.
- `fit_predict_model`: Trains an XGBoost model and generates predictions iteratively.
- `reconstruct_actuals`: Reconstructs actual values from differenced predictions.
- `save_predictions_to_db`: Stores prediction results in a MongoDB database.
- `save_model`: Saves the trained model as a pickle file.
- `save_feature_importances`: Logs and saves top feature importances.
- `train_model`: Orchestrates the end-to-end training process.

"""
import json
import sys, os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.configs.logger_config import logger
from streamlit_app.components.forecaster import Forecaster

def prepare_data(original_data, best_chain, variable):
    """
    Prepares training and testing datasets by differencing the data and applying feature engineering.

    Args:
        original_data (pd.DataFrame): The original time-series dataset.
        best_chain (list): Ordered list of variables representing the causal chain.
        variable (str): Target variable for prediction.

    Returns:
        tuple: Split datasets (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If 'FEDFUNDS' column is not found in the dataset.
    """
    X = original_data[best_chain[:best_chain.index(variable)]]
    y = original_data[variable]

    X_diff = X.diff().dropna()
    y_diff = y.diff().dropna()

    if 'FEDFUNDS' not in X_diff.columns:
        raise ValueError("FEDFUNDS column not found in diff_df")

    first_fed_funds = original_data['FEDFUNDS'].iloc[0] if 'FEDFUNDS' in original_data.columns else 0
    X_diff['FEDFUNDS_Actual'] = X_diff['FEDFUNDS'].cumsum() + first_fed_funds

    X_diff = Forecaster.add_engineered_features(X_diff)
    features_to_fill = [
        'FEDFUNDS_pct_change', 'FEDFUNDS_Duration', 'Time_Since_Change',
        'FEDFUNDS_interaction', 'FEDFUNDS_Deviation', 'FEDFUNDS_Direction',
        'FEDFUNDS_Absolute_Change', 'FEDFUNDS_Rolling_Volatility'
    ]
    X_diff[features_to_fill] = X_diff[features_to_fill].fillna(0)
    # # For lag features, initial NaNs are expected; decide whether to fill or drop
    # # Here, we'll drop rows with NaNs resulting from lagging
    X_diff.dropna(inplace=True)

    y_diff = y_diff.loc[X_diff.index]

    train_size = int(len(X_diff) * 0.8)
    X_train, X_test = X_diff[:train_size], X_diff[train_size:]
    y_train, y_test = y_diff[:train_size], y_diff[train_size:]

    logger.info(f"Data prepared for {variable} with train size: {train_size}")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, X_test, y_train, y_test

def fit_predict_model(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost model incrementally and makes predictions on the test set.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target variable.

    Returns:
        tuple: Trained XGBoost model, predictions for the test set, and MAE over time.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=1
    )

    predictions = []
    mae_over_time = []  # To store MAE over time

    for i in range(len(X_test)):
        model.fit(X_train, y_train)
        pred = model.predict(X_test.iloc[i:i+1])
        predictions.append(pred[0])

        # Calculate MAE for the current prediction
        current_mae = mean_absolute_error(y_test.iloc[:i+1], predictions)
        mae_over_time.append(current_mae)

        X_train = pd.concat([X_train, X_test.iloc[i:i+1]])
        y_train = pd.concat([y_train, pd.Series(y_test.iloc[i], index=[X_test.index[i]])])

    return model, predictions, mae_over_time


def reconstruct_actuals(original_data, variable, train_size, predictions, X_test):
    """
     Reconstructs the original scale of the target variable from differenced predictions.

     Args:
         original_data (pd.DataFrame): The original dataset.
         variable (str): Target variable for prediction.
         train_size (int): Number of training samples.
         predictions (list): Differenced predictions.
         X_test (pd.DataFrame): Testing features.

     Returns:
         dict: Reconstructed actual values for the test set.
     """
    last_actual_value = original_data[variable].iloc[train_size]
    real_predictions = {}
    cumulative_sum = last_actual_value

    for date, diff_value in zip(X_test.index, predictions):
        cumulative_sum += diff_value
        real_predictions[date] = cumulative_sum

    logger.info(f"Actual values reconstructed for {variable}.")
    return real_predictions

def save_predictions_to_db(variable, real_predictions, mae_over_time):
    """
    Saves reconstructed predictions and MAE values to a MongoDB database.

    Args:
        variable (str): Target variable for prediction.
        real_predictions (dict): Reconstructed predictions.
        mae_over_time (list): MAE values over time.
    """
    try:
        # Convert predictions and MAE to a format suitable for database storage
        data_to_save = {
            date: {
                f'{variable}_prediction': pred_value,
                'mae_over_time': mae_value
            }
            for (date, pred_value), mae_value in zip(real_predictions.items(), mae_over_time)
        }

        with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")) as loader:
            loader.insert_predictions_data(variable, data_to_save)
        logger.info(f"Predictions and MAE values for {variable} saved to database.")
    except Exception as e:
        logger.error(f"Failed to save predictions for {variable}: {e}")

def save_model(model, variable):
    """
    Saves the trained model to a pickle file.

    Args:
        model (xgb.XGBRegressor): Trained XGBoost model.
        variable (str): Target variable for prediction.
    """
    os.makedirs(os.path.join(os.getcwd(), "src/resources/models"), exist_ok=True)
    model_filename = os.path.join("src/resources/models", f"best_model_{variable}.pkl")
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    logger.info(f"Model for {variable} saved as {model_filename}")

def save_feature_importances(model, X_train, variable, feature_importances):
    """
    Logs and saves the top 10 feature importances for the trained model.

    Args:
        model (xgb.XGBRegressor): Trained XGBoost model.
        X_train (pd.DataFrame): Training features.
        variable (str): Target variable for prediction.
        feature_importances (dict): Dictionary to store feature importance values.
    """
    importances = dict(zip(X_train.columns, model.feature_importances_))
    # Convert numpy.float32 to native Python float
    importances = {k: float(v) for k, v in importances.items()}
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    feature_importances[variable] = dict(sorted_importances)
    logger.info(f"Top 10 feature importances for {variable}: {feature_importances[variable]}")

def calculate_metrics(y_true, y_pred):
    """
    Calculate various performance metrics for model evaluation.

    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values

    Returns:
        dict: Dictionary containing various performance metrics
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def train_model(best_chain, original_data):
    """
    Executes the end-to-end training process for variables in the causal chain.

    Args:
        best_chain (list): Ordered list of variables representing the causal chain.
        original_data (pd.DataFrame): The original time-series dataset.

    Saves:
        - Predictions to MongoDB.
        - Models as pickle files.
        - Feature importances as a JSON file.
    """
    feature_importances = {}
    model_metrics = {}

    for variable in best_chain[1:]:
        logger.info(f"Starting model training for {variable}")

        # 1. Prepare data
        X_train, X_test, y_train, y_test = prepare_data(original_data, best_chain, variable)

        # 2. Fit model and make predictions
        model, predictions, mae_over_time = fit_predict_model(X_train, y_train, X_test, y_test)

        # 3. Calculate and log Mean Absolute Error
        mae_test = mean_absolute_error(y_test, predictions)
        logger.info(f"Mean Absolute Error for {variable}: {mae_test}")

        # 4. Reconstruct actual values from differenced predictions
        real_predictions = reconstruct_actuals(original_data, variable, len(X_train), predictions, X_test)

        # 5. Save predictions, model, and feature importances
        save_predictions_to_db(variable, real_predictions, mae_over_time)
        save_model(model, variable)
        save_feature_importances(model, X_train, variable, feature_importances)

        # Calculate metrics
        diff_metrics = calculate_metrics(y_test, predictions)
        actual_values = original_data[variable].loc[X_test.index]
        pred_values = pd.Series(real_predictions).loc[X_test.index]
        actual_metrics = calculate_metrics(actual_values, pred_values)

        model_metrics[variable] = {
            'differenced': diff_metrics,
            'actual': actual_metrics
        }

    # Save feature importances and metrics
    with open('src/resources/models/feature_importances.json', 'w') as f:
        json.dump(feature_importances, f, indent=4)

    with open('src/resources/models/model_metrics.json', 'w') as f:
        json.dump(model_metrics, f, indent=4)

