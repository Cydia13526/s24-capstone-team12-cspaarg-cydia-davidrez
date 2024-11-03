import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.configs.logger_config import logger

class Forecaster:
    def __init__(self, models, data, historical_df, forecasted_fedfunds, variables, target):
        self.models = models
        self.data = data
        self.variables = variables  
        self.historical_df = historical_df
        self.forecasted_fedfunds = forecasted_fedfunds
        self.target = target

    @staticmethod
    def add_engineered_features(diff_df: pd.DataFrame) -> pd.DataFrame:
        if 'FEDFUNDS' not in diff_df.columns:
            raise ValueError("FEDFUNDS column not found in diff_df")

        # 1. Calculate FEDFUNDS percentage change
        diff_df['FEDFUNDS_pct_change'] = diff_df['FEDFUNDS_Actual'].replace(0, np.nan).pct_change() * 100

        # 2. Calculate duration of constant FEDFUNDS values
        diff_df['FEDFUNDS_Duration'] = (
                diff_df['FEDFUNDS_Actual']
                .ne(diff_df['FEDFUNDS_Actual'].shift())
                .cumsum()
                .groupby(diff_df['FEDFUNDS_Actual'].ne(diff_df['FEDFUNDS_Actual'].shift()).cumsum())
                .cumcount() + 1
        )

        # 3. Compute Exponentially Weighted Moving Average
        diff_df['FEDFUNDS_Weighted'] = diff_df['FEDFUNDS_Actual'].ewm(span=12, adjust=False).mean()

        # 4. Calculate time since the last change in FEDFUNDS
        diff_df['Time_Since_Change'] = (
                diff_df['FEDFUNDS']
                .eq(0).cumsum()
                - diff_df['FEDFUNDS'].eq(0).cumsum().where(diff_df['FEDFUNDS'].ne(0)).ffill().fillna(0).astype(int)
        )

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

        # 5. Create interaction terms and other derived features
        diff_df['FEDFUNDS_interaction'] = diff_df['FEDFUNDS_Actual'] * diff_df['FEDFUNDS_Duration']
        diff_df['FEDFUNDS_Expanding_Mean'] = diff_df['FEDFUNDS_Actual'].expanding().mean()
        diff_df['FEDFUNDS_Deviation'] = diff_df['FEDFUNDS_Actual'] - diff_df['FEDFUNDS_Expanding_Mean']

        # 6. Calculate rolling statistics (mean and standard deviation)
        diff_df['FEDFUNDS_Rolling_mean'] = diff_df['FEDFUNDS_Actual'].rolling(window=12, min_periods=1).mean()
        diff_df['FEDFUNDS_Rolling_std'] = diff_df['FEDFUNDS_Actual'].rolling(window=12, min_periods=1).std()

        # 7. Create lag features for FEDFUNDS
        for lag in range(1, 13):
            diff_df[f'FEDFUNDS_lag_{lag}'] = diff_df['FEDFUNDS'].shift(lag)

        # 8. Additional engineered features
        diff_df['FEDFUNDS_Direction'] = diff_df['FEDFUNDS'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        diff_df['FEDFUNDS_Absolute_Change'] = diff_df['FEDFUNDS'].abs()
        diff_df['FEDFUNDS_Rolling_Volatility'] = diff_df['FEDFUNDS'].rolling(window=12, min_periods=1).std()

        # # Yield Spread (if 'DGS10' and 'DGS2' are available in diff_df)
        # if 'DGS10' in diff_df.columns and 'DGS2' in diff_df.columns:
        #     diff_df['Yield_Spread'] = diff_df['DGS10'] - diff_df['DGS2']

        # 9. Interaction with other variables, if available
        if 'UNRATE' in diff_df.columns:
            diff_df['FEDFUNDS_UNRATE_interaction'] = diff_df['FEDFUNDS'] * diff_df['UNRATE']

        # 10. Handle missing values in calculated features
        features_to_fill = [
            'FEDFUNDS_pct_change', 'FEDFUNDS_Duration', 'Time_Since_Change',
            'FEDFUNDS_interaction', 'FEDFUNDS_Deviation', 'FEDFUNDS_Direction',
            'FEDFUNDS_Absolute_Change', 'FEDFUNDS_Rolling_Volatility'
        ]
        diff_df[features_to_fill] = diff_df[features_to_fill].fillna(0)
        # For lag features, initial NaNs are expected; decide whether to fill or drop
        # Here, we'll drop rows with NaNs resulting from lagging
        diff_df.dropna(inplace=True)
        return diff_df

    def prepare_features(self, current_forecast_df: pd.DataFrame, variable: str) -> pd.DataFrame:
        features = self.fedfund_features(current_forecast_df)
        for prev_var in self.variables[1:self.variables.index(variable)]:
            if prev_var in current_forecast_df.columns:
                features.append(prev_var)

        X_forecast = current_forecast_df[features]
        logger.info(f"X_forecast shape: {X_forecast.shape}")
        logger.info(f"X_forecast columns: {X_forecast.columns}")

        required_features = self.models[variable].get_booster().feature_names
        for feat in required_features:
            if feat not in X_forecast.columns:
                X_forecast[feat] = 0

        X_forecast = X_forecast[required_features]
        return X_forecast

    @staticmethod
    def fedfund_features(df):
        features = [
            'FEDFUNDS', 'FEDFUNDS_Actual', 'FEDFUNDS_pct_change',
            'FEDFUNDS_Duration', 'FEDFUNDS_Weighted',
            'Time_Since_Change',
            'FEDFUNDS_interaction', 'FEDFUNDS_Deviation',
            'FEDFUNDS_Expanding_Mean',
            'FEDFUNDS_Rolling_mean', 'FEDFUNDS_Rolling_std',
            'FEDFUNDS_Direction', 'FEDFUNDS_Absolute_Change',
            'FEDFUNDS_Rolling_Volatility'
        ] + [f'FEDFUNDS_lag_{lag}' for lag in range(1, 13)]
        # + [f'FEDFUNDS_Actual_lag_{lag}' for lag in range(1, 13)]

        # if 'DGS10' in df.columns and 'DGS2' in df.columns:
        #     features.append('Yield_Spread')

        if 'UNRATE' in df.columns:
            features.append('FEDFUNDS_UNRATE_interaction')
        return features
    
    def forecast(self, models, data, target, data_path, model_path):
        forecast_df = pd.DataFrame({'FEDFUNDS_Actual': self.forecasted_fedfunds})
        forecasts = {'FEDFUNDS': self.forecasted_fedfunds}

        logger.info(f"Initial forecast_df shape: {forecast_df.shape}")
        logger.info(f"Initial forecast_df index: {forecast_df.index}")

        for variable in self.variables[1:]:
            logger.info(f"Forecasting {variable}")

            diff_df = forecast_df.copy()
            diff_df['FEDFUNDS_Actual'] = self.forecasted_fedfunds
            last_historical_fedfunds = self.historical_df['FEDFUNDS'].iloc[-1]
            diff_df['FEDFUNDS'] = self.forecasted_fedfunds.diff().fillna(self.forecasted_fedfunds.iloc[0] - last_historical_fedfunds)
            current_forecast_df = self.add_engineered_features(diff_df)

            logger.info(f"current_forecast_df shape after adding features: {current_forecast_df.shape}")
            logger.info(f"current_forecast_df columns: {current_forecast_df.columns}")
            
            if current_forecast_df.empty:
                logger.info(f"Warning: No data available for forecasting {variable}")
                forecasts[variable] = pd.Series(index=forecast_df.index)
                continue

            X_forecast = self.prepare_features(current_forecast_df, variable)

            model = models[variable]
            forecast_diff = model.predict(X_forecast)

            logger.info(f"forecast_diff shape: {forecast_diff.shape}")
            logger.info(f"forecast_diff: {forecast_diff}")

            last_actual = data[variable].iloc[-1]
            forecast_original = last_actual + np.cumsum(forecast_diff)
            forecasts[variable] = pd.Series(forecast_original, index=forecast_df.index)
            forecast_df[variable] = forecast_diff

            logger.info(f"forecast_original shape: {forecast_original.shape}")
            logger.info(f"forecast_original: {forecast_original}")
            logger.info(f"Model used for {variable}:")
            logger.info(model)
            logger.info(f"Features used for {variable}:")
            logger.info(X_forecast.columns)

            with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")) as loader:
                loader.insert_forecast_data(variable, forecast_original)
        return forecasts