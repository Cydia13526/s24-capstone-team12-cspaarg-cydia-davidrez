from typing import List
import pandas as pd
import numpy as np
import joblib
import sqlite3
import warnings

from scipy.fftpack import diff
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

class Forecaster:
    def __init__(self, models, data, historical_df, forecasted_fedfunds, variables, target):
        self.models = models
        self.data = data
        self.variables = variables  
        self.historical_df = historical_df
        self.forecasted_fedfunds = forecasted_fedfunds
        self.target = target

    @staticmethod
    def add_engineered_features(diff_df: pd.DataFrame, forecasted_fedfunds: pd.Series, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features based on FEDFUNDS to the differenced DataFrame.

        Parameters
        ----------
        diff_df : pd.DataFrame
            DataFrame to add features to.
        forecasted_fedfunds : pd.Series
            Series containing the forecasted FEDFUNDS values.
        historical_df : pd.DataFrame
            DataFrame containing historical data, including FEDFUNDS.

        Returns
        -------
        diff_df : pd.DataFrame
            DataFrame with additional engineered features.
        """

        # 1. Reconstruct FEDFUNDS actual values from forecasted data
        diff_df['FEDFUNDS_Actual'] = forecasted_fedfunds

        # 2. Calculate the differenced FEDFUNDS
        last_historical_fedfunds = historical_df['FEDFUNDS'].iloc[-1]
        diff_df['FEDFUNDS'] = forecasted_fedfunds.diff().fillna(forecasted_fedfunds.iloc[0] - last_historical_fedfunds)
        
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
        # Instead of dropping NaN values, fill them
        diff_df.fillna(0, inplace=True)

        return diff_df

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
        # Create a DataFrame with the forecasted FEDFUNDS
        forecast_df = pd.DataFrame({'FEDFUNDS_Actual': self.forecasted_fedfunds})
        
        print(f"Initial forecast_df shape: {forecast_df.shape}")
        print(f"Initial forecast_df index: {forecast_df.index}")
        
        forecasts = {'FEDFUNDS': self.forecasted_fedfunds}
        
        # 3. Loop over the best_chain variables (excluding FEDFUNDS)
        for variable in self.variables[1:]:
            print(f"Forecasting {variable}")
            
            # Add engineered features for the current variable
            current_forecast_df = self.add_engineered_features(forecast_df.copy(), self.forecasted_fedfunds, self.historical_df)
            
            print(f"current_forecast_df shape after adding features: {current_forecast_df.shape}")
            print(f"current_forecast_df columns: {current_forecast_df.columns}")
            
            if current_forecast_df.empty:
                print(f"Warning: No data available for forecasting {variable}")
                forecasts[variable] = pd.Series(index=forecast_df.index)
                continue

            # 4. Load the appropriate model for the variable
            model = models[variable]
            
            # Prepare features for the current variable
            features = self.fedfund_features(current_forecast_df)
            for prev_var in self.variables[1:self.variables.index(variable)]:
                if prev_var in current_forecast_df.columns:
                    features.append(prev_var)
            
            X_forecast = current_forecast_df[features]
            
            print(f"X_forecast shape: {X_forecast.shape}")
            print(f"X_forecast columns: {X_forecast.columns}")
            
            # Ensure all required features are present
            required_features = model.get_booster().feature_names
            for feat in required_features:
                if feat not in X_forecast.columns:
                    print(f"Warning: {feat} not in features. Adding it with zeros.")
                    X_forecast[feat] = 0
            
            # Ensure features are in the correct order
            X_forecast = X_forecast[required_features]
            
            # 5. Make prediction (in differenced form)
            forecast_diff = model.predict(X_forecast)
            
            print(f"forecast_diff shape: {forecast_diff.shape}")
            print(f"forecast_diff: {forecast_diff}")
            
            # 6. Convert back to original scale
            last_actual = data[variable].iloc[-1]
            forecast_original = last_actual + np.cumsum(forecast_diff)
            
            print(f"forecast_original shape: {forecast_original.shape}")
            print(f"forecast_original: {forecast_original}")
            
            # Add the forecast to the results and to the forecast_df for the next iteration
            forecasts[variable] = pd.Series(forecast_original, index=forecast_df.index)
            forecast_df[variable] = forecast_diff
            
            print(f"Model used for {variable}:")
            print(model)
            print(f"Features used for {variable}:")
            print(X_forecast.columns)
            
            # 7. Store the forecast in the database
            conn = sqlite3.connect(data_path + 'forecast_macro.db')
            cursor = conn.cursor()
            
            for date, value in zip(forecast_df.index, forecast_original):
                query = f"""
                INSERT OR REPLACE INTO forecast_macro (date, model, {variable}_forecast)
                VALUES (?, ?, ?)
                """
                cursor.execute(query, (date.strftime('%Y-%m-%d'), 'user_forecast', float(value)))
            
            conn.commit()
            conn.close()
            
            print(f"Forecast for {variable} has been added to the database.")
        
        return forecasts