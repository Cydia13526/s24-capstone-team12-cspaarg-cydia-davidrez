import json
import sys, os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.configs.logger_config import logger
from streamlit_app.components.forecaster import Forecaster

def prepare_data(original_data, best_chain, variable):
    X = original_data[best_chain[:best_chain.index(variable)]]
    y = original_data[variable]

    X_diff = X.diff().dropna()
    y_diff = y.diff().dropna()

    if 'FEDFUNDS' not in X_diff.columns:
        raise ValueError("FEDFUNDS column not found in diff_df")

    first_fed_funds = original_data['FEDFUNDS'].iloc[0] if 'FEDFUNDS' in original_data.columns else 0
    X_diff['FEDFUNDS_Actual'] = X_diff['FEDFUNDS'].cumsum() + first_fed_funds

    X_diff = Forecaster.add_engineered_features(X_diff)
    y_diff = y_diff.loc[X_diff.index]

    train_size = int(len(X_diff) * 0.8)
    X_train, X_test = X_diff[:train_size], X_diff[train_size:]
    y_train, y_test = y_diff[:train_size], y_diff[train_size:]

    logger.info(f"Data prepared for {variable} with train size: {train_size}")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, X_test, y_train, y_test

def fit_predict_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=1000
    )

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.info(f"X_train index range: {X_train.index.min()} to {X_train.index.max()}")
    logger.info(f"X_test index range: {X_test.index.min()} to {X_test.index.max()}")
    logger.info(f"features: {X_train.columns}")

    predictions = []
    for i in range(len(X_test)):
        model.fit(X_train, y_train)
        pred = model.predict(X_test.iloc[i:i+1])
        predictions.append(pred[0])

        X_train = pd.concat([X_train, X_test.iloc[i:i+1]])
        y_train = pd.concat([y_train, pd.Series(y_test.iloc[i], index=[X_test.index[i]])])

    logger.info(f"Model training completed for variable with final train shape: {X_train.shape}")
    return model, predictions


def reconstruct_actuals(original_data, variable, train_size, predictions, X_test):
    last_actual_value = original_data[variable].iloc[train_size]
    real_predictions = {}
    cumulative_sum = last_actual_value

    for date, diff_value in zip(X_test.index, predictions):
        cumulative_sum += diff_value
        real_predictions[date] = cumulative_sum

    logger.info(f"Actual values reconstructed for {variable}.")
    return real_predictions

def save_predictions_to_db(variable, real_predictions):
    try:
        with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")) as loader:
            loader.insert_predictions_data(variable, real_predictions)
        logger.info(f"Predictions for {variable} saved to database.")
    except Exception as e:
        logger.error(f"Failed to save predictions for {variable}: {e}")

def save_model(model, variable):
    os.makedirs(os.path.join(os.getcwd(), "src/resources/models"), exist_ok=True)
    model_filename = os.path.join("src/resources/models", f"best_model_{variable}.pkl")
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    logger.info(f"Model for {variable} saved as {model_filename}")

def save_feature_importances(model, X_train, variable, feature_importances):
    importances = dict(zip(X_train.columns, model.feature_importances_))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    feature_importances[variable] = dict(sorted_importances)
    logger.info(f"Top 10 feature importances for {variable}: {feature_importances[variable]}")

def train_model(best_chain, original_data):
    feature_importances = {}

    for variable in best_chain[1:]:
        logger.info(f"Starting model training for {variable}")

        # 1. Prepare data
        X_train, X_test, y_train, y_test = prepare_data(original_data, best_chain, variable)

        # 2. Fit model and make predictions
        model, predictions = fit_predict_model(X_train, y_train, X_test, y_test)

        # 3. Calculate and log Mean Absolute Error
        mae_test = mean_absolute_error(y_test, predictions)
        logger.info(f"Mean Absolute Error for {variable}: {mae_test}")

        # 4. Reconstruct actual values from differenced predictions
        real_predictions = reconstruct_actuals(original_data, variable, len(X_train), predictions, X_test)

        # 5. Save predictions, model, and feature importances
        save_predictions_to_db(variable, real_predictions)
        save_model(model, variable)
        save_feature_importances(model, X_train, variable, feature_importances)

    # 6. Save all feature importances to a JSON file
    with open('resources/models/feature_importances.json', 'w') as f:
        json.dump(feature_importances, f, indent=4)
    logger.info("Feature importances saved to resources/models/feature_importances.json.")
