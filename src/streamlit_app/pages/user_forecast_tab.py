"""
This module provides utilities for visualizing and forecasting macroeconomic indicators,
focusing on the Federal Funds Rate and its impacts on other macroeconomic variables.

The functionalities include:
- Plotting Federal Funds Rate forecasts.
- Displaying selected macroeconomic variable forecasts.
- Generating multi-line plots for historical and forecasted data.
- Creating, adjusting, and storing forecasts.
- Session state management for Streamlit applications.

Functions:
    - plot_fedfunds_forecast: Plots a forecast of the Federal Funds Rate.
    - plot_selected_forecasts: Displays selected macroeconomic variable forecasts.
    - plot_forecasts: Visualizes forecasts for multiple variables as subplots.
    - plot_multi_line: Creates a multi-line plot for multiple variables over time.
    - create_fedfunds_forecast: Generates a Federal Funds Rate forecast based on input changes.
    - forecast_macro_variables: Predicts macroeconomic variables using trained models.
    - store_forecasts: Stores generated forecasts in a MongoDB database.
    - generate_forecasts: Orchestrates the generation and storage of forecasts.
    - initialize_session_state_scenario: Initializes session state variables for a Streamlit scenario.
    - adjust_fed_funds_rate: Provides Streamlit UI for adjusting Federal Funds Rate changes.
    - forecast_macros: Generates macroeconomic forecasts for a given scenario.
    - display_forecast_results: Displays forecast results interactively in a Streamlit app.
"""

import sys, os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.components.forecaster import Forecaster
from streamlit_app.utils.common_util import add_recession_highlights
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.configs.logger_config import logger
from streamlit_app.data.fred_data_loader import variable_descriptions

def plot_fedfunds_forecast(forecast, start_date, current_month):
    """
    Plots the Federal Funds Rate forecast over time with highlights for the current forecast month.

    Parameters:
        forecast (list of float): List of Federal Funds Rate values.
        start_date (str): Start date of the forecast in 'YYYY-MM-DD' format.
        current_month (int): Index of the current forecast month.

    Returns:
        plotly.graph_objects.Figure: Plotly figure of the forecast.
    """
    date_range = pd.date_range(start=start_date, periods=len(forecast), freq='MS')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_range,
        y=forecast,
        mode='lines+markers',
        name='Federal Funds Rate',
        line=dict(color='blue'),
        marker=dict(size=8, color='blue'),
    ))

    y_min = min(forecast) - 0.5
    y_max = max(forecast) + 0.5
    fig.update_layout(yaxis_range=[y_min, y_max])

    if current_month < len(date_range):
        fig.add_vrect(
            x0=date_range[current_month],
            x1=date_range[current_month + 1] if current_month + 1 < len(date_range) else date_range[-1],
            fillcolor="LightGreen", opacity=0.5,
            layer="below", line_width=0,
        )
        fig.add_annotation(
            x=date_range[current_month],
            y=y_max,
            text="Current Forecast Month",
            showarrow=False,
            yshift=10
        )

    fig.update_layout(
        title='Federal Funds Rate Forecast',
        xaxis_title='Date',
        yaxis_title='Federal Funds Rate (%)',
        height=400,
        hovermode='x unified'
    )
    fig.update_xaxes(tickformat='%Y-%m-%d')
    return fig

def plot_selected_forecasts(historical_df, forecast_df, selected_variables, start_date, end_date):
    """
    Displays historical and forecasted data for selected macroeconomic variables.

    Parameters:
        historical_df (pd.DataFrame): Historical data indexed by date.
        forecast_df (dict): Forecasted data, with variables as keys and lists as values.
        selected_variables (list of str): Variables to include in the plot.
        start_date (str): Start date for the plot.
        end_date (str): End date for the plot.

    Returns:
        plotly.graph_objects.Figure: Plotly figure of the selected forecasts.
    """
    fig = go.Figure()

    h_df = historical_df.copy()
    h_df.rename(columns=variable_descriptions, inplace=True)

    f_df = forecast_df.copy()
    f_df = {variable_descriptions.get(k, k): v for k, v in f_df.items()}

    for variable in selected_variables:
        fig.add_trace(go.Scatter(
            x=h_df.index,
            y=h_df[variable],
            mode='lines',
            name=f"{variable} Historical"
        ))

        forecast_dates = pd.date_range(start=h_df.index[-1] + pd.DateOffset(months=1), periods=len(f_df[variable]), freq='MS')
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=f_df[variable],
            mode='lines',
            name=f"{variable} Forecast"
        ))

    fig.update_layout(
        height=600,
        title_text="Macro Variable Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Data Type",
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            range=[start_date, end_date]
        )
    )

    fig = add_recession_highlights(fig)  # Add this line

    return fig

def plot_forecasts(historical_data, forecasts):
    """
    Plots historical data and forecasts for multiple macroeconomic variables.

    Args:
        historical_data (pd.DataFrame): Historical time-series data indexed by date.
        forecasts (dict): A dictionary where keys are variable names and values are forecasted values.

    Returns:
        plotly.graph_objects.Figure: A multi-panel Plotly figure visualizing historical data and forecasts.
    """
    fig = make_subplots(rows=len(forecasts), cols=1, subplot_titles=list(forecasts.keys()))

    for i, (variable, forecast) in enumerate(forecasts.items(), start=1):
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data[variable], name=f"{variable} Historical"),
            row=i, col=1
        )

        forecast_dates = pd.date_range(start=historical_data.index[-1] + pd.DateOffset(months=1), periods=len(forecast), freq='MS')
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=forecast, name=f"{variable} Forecast"),
            row=i, col=1
        )

    fig.update_layout(height=300*len(forecasts), title_text="Macro Variable Forecasts")
    for i in range(1, len(forecasts) + 1):
        fig.update_xaxes(
            title_text="Date",
            type='date',
            tickformat='%Y-%m-%d',
            row=i, col=1
        )
        fig.update_yaxes(title_text="Value", row=i, col=1)

    fig = add_recession_highlights(fig)  # Add this line

    return fig

def plot_multi_line(df, title, variable_descriptions):
    """
      Creates a line plot for multiple variables in a DataFrame.

      Args:
          df (pd.DataFrame): DataFrame with time-series data for multiple variables indexed by date.
          title (str): Title of the plot.
          variable_descriptions (list): Descriptions for each variable in the DataFrame.

      Returns:
          plotly.graph_objects.Figure: A line plot with interactive features like a range slider.
      """
    fig = go.Figure()
    for column, description in zip(df.columns, variable_descriptions):
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=description))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            range=[df.index.min(), df.index.max()]
        )
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig = add_recession_highlights(fig)
    return fig

def create_fedfunds_forecast(forecast_months, start_value, changes):
    """
      Generates a Federal Funds Rate forecast based on user-provided changes.

      Args:
          forecast_months (int): Number of months for the forecast.
          start_value (float): Initial Federal Funds Rate.
          changes (list): List of changes in basis points for each period.

      Returns:
          list: A list representing the Federal Funds Rate forecast.
      """
    forecast = [start_value]
    for change in changes:
        forecast.append(forecast[-1] + change / 100)  # Convert basis points to percentage
    while len(forecast) < forecast_months + 1:
        forecast.append(forecast[-1])
    return forecast[:forecast_months + 1]

def forecast_macro_variables(fedfunds_forecast, original_data, differenced_data, best_chain, models):
    """
    Forecasts macroeconomic variables based on Federal Funds Rate forecasts.

    Args:
        fedfunds_forecast (pd.Series): Federal Funds Rate forecast.
        original_data (pd.DataFrame): Original macroeconomic data.
        differenced_data (pd.DataFrame): Differenced macroeconomic data for stationarity.
        best_chain (list): Order of variable dependencies for forecasting.
        models (dict): A dictionary of trained models for forecasting.

    Returns:
        dict: Forecasts for all macroeconomic variables.
    """
    forecast_df = pd.DataFrame({'FEDFUNDS': fedfunds_forecast}, index=fedfunds_forecast.index)
    forecast_df['FEDFUNDS'] = forecast_df['FEDFUNDS'].diff().fillna(forecast_df['FEDFUNDS'].iloc[0] - original_data['FEDFUNDS'].iloc[-1])
    diff_df = forecast_df.copy()

    forecast_df = Forecaster.add_engineered_features(diff_df)
    features_to_fill = [
        'FEDFUNDS_pct_change', 'FEDFUNDS_Duration', 'Time_Since_Change',
        'FEDFUNDS_interaction', 'FEDFUNDS_Deviation', 'FEDFUNDS_Direction',
        'FEDFUNDS_Absolute_Change', 'FEDFUNDS_Rolling_Volatility'
    ]
    forecast_df[features_to_fill] = forecast_df[features_to_fill].fillna(0)
    # # For lag features, initial NaNs are expected; decide whether to fill or drop
    # # Here, we'll drop rows with NaNs resulting from lagging
    forecast_df.dropna(inplace=True)

    forecasts = {'FEDFUNDS': fedfunds_forecast}

    for variable in best_chain[1:]:
        model = models[variable]
        features = Forecaster.fedfund_features()
        X_forecast = forecast_df[features]
        y_pred = model.predict(X_forecast)
        last_actual = original_data[variable].iloc[-1]
        forecasts[variable] = last_actual + np.cumsum(y_pred)
        forecast_df[variable] = y_pred
    return forecasts

def store_forecasts(forecasts, original_data):
    """
    Stores macroeconomic forecasts into a MongoDB database.

    Args:
        forecasts (dict): A dictionary of forecasted values for macroeconomic variables.
        original_data (pd.DataFrame): Original macroeconomic data to infer forecast dates.

    Raises:
        Exception: Logs an error if the forecast data cannot be saved.
    """
    try:
        last_date = original_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(next(iter(forecasts.values()))))
        with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/forecast_macro.db")) as loader:
            for variable in forecasts:
                if not isinstance(forecasts[variable], pd.Series):
                    logger.info(f"Warning: forecast for {variable} is not a pandas Series. Skipping.")
                    continue
                loader.insert_forecast_data(variable, zip(forecast_dates, forecasts[variable]))
    except Exception as e:
        logger.error(f"Failed to save predictions for {variable}: {e}")

def generate_forecasts(original_data, best_chain, models, forecast_months, fedfunds_forecast):
    """
      Generates and stores forecasts for macroeconomic variables.

      Args:
          original_data (pd.DataFrame): Historical macroeconomic data.
          best_chain (list): Order of variable dependencies for forecasting.
          models (dict): A dictionary of trained models for forecasting.
          forecast_months (int): Number of months to forecast.
          fedfunds_forecast (list): Federal Funds Rate forecast.

      Returns:
          dict: Forecasts for macroeconomic variables.
      """
    forecast_dates = pd.date_range(start=original_data.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    fed_funds_forecast_series = pd.Series(fedfunds_forecast, index=forecast_dates)

    forecaster = Forecaster(models, original_data, fed_funds_forecast_series, best_chain, 'FEDFUNDS')
    forecasts = forecaster.forecast(models, original_data, 'FEDFUNDS', 'data/', 'models/')
    store_forecasts(forecasts, original_data)
    return forecasts

def initialize_session_state_scenario(scenario_number, original_data):
    """
    Initializes Streamlit session state variables for a forecast scenario.

    Args:
        scenario_number (int): Unique identifier for the scenario.
        original_data (pd.DataFrame): Historical macroeconomic data.
    """
    if f'forecast_months_{scenario_number}' not in st.session_state:
        st.session_state[f'forecast_months_{scenario_number}'] = 12
    if f'start_value_{scenario_number}' not in st.session_state:
        st.session_state[f'start_value_{scenario_number}'] = original_data['FEDFUNDS'].iloc[-1]
    if f'changes_{scenario_number}' not in st.session_state:
        st.session_state[f'changes_{scenario_number}'] = []
    if f'forecasts_{scenario_number}' not in st.session_state:
        st.session_state[f'forecasts_{scenario_number}'] = None
    if f'current_month_{scenario_number}' not in st.session_state:
        st.session_state[f'current_month_{scenario_number}'] = 0

def adjust_fed_funds_rate(scenario_number):
    """
    Handles user inputs to adjust the Federal Funds Rate in a Streamlit application.

    Args:
        scenario_number (int): Unique identifier for the scenario.
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        change = st.selectbox("Change", ["-50 bps", "-25 bps", "No change", "+25 bps", "+50 bps"], index=2, key=f"fed_fund_rate_select_box{scenario_number}")
    with col2:
        months = st.number_input("Month(s) Applied", min_value=1, max_value=st.session_state[f'forecast_months_{scenario_number}'] - st.session_state[f'current_month_{scenario_number}'] , value=1, key=f"fed_fund_rate_number_input{scenario_number}")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply Change", key=f"apply_change_{scenario_number}"):
            change_value = int(change.split()[0]) if change != "No change" else 0
            st.session_state[f'changes_{scenario_number}'].extend([change_value] * months)
            st.session_state[f'current_month_{scenario_number}'] += months
            if st.session_state[f'current_month_{scenario_number}'] >= st.session_state[f'forecast_months_{scenario_number}']:
                st.session_state[f'current_month_{scenario_number}'] = st.session_state[f'forecast_months_{scenario_number}'] - 1
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"Reset Forecast", key=f"reset_forecast_{scenario_number}"):
            st.session_state[f"changes_{scenario_number}"] = []
            st.session_state[f"current_month_{scenario_number}"] = 0

def forecast_macros(fed_funds_forecast, original_data, models, best_chain, start_date, scenario_number):
    """
    Forecasts macroeconomic variables and stores results for a specific scenario.

    Args:
        fed_funds_forecast (list): Federal Funds Rate forecast.
        original_data (pd.DataFrame): Historical macroeconomic data.
        models (dict): A dictionary of trained models for forecasting.
        best_chain (list): Order of variable dependencies for forecasting.
        start_date (pd.Timestamp): Start date of the forecast.
        scenario_number (int): Unique identifier for the scenario.
    """
    fed_funds_forecast_series = pd.Series(fed_funds_forecast[1:], index=pd.date_range(start=start_date, periods=st.session_state[f'forecast_months_{scenario_number}'], freq='MS'))

    logger.info(f"fed_funds_forecast_series: {fed_funds_forecast_series}")
    forecaster = Forecaster(models, original_data, original_data, fed_funds_forecast_series, best_chain, 'FEDFUNDS')
    st.session_state[f'forecasts_{scenario_number}'] = forecaster.forecast(models, original_data, 'FEDFUNDS', 'src/resources/data/mongo_db/', 'models/')

    # logger.info("Forecasts generated:")
    # for var, forecast in st.session_state[f'forecasts_{scenario_number}'].items():
    #     logger.info(f"{var}: {forecast}")

    store_forecasts(st.session_state[f'forecasts_{scenario_number}'], original_data)

def display_forecast_results(original_data, best_chain, scenario_number):
    """
    Displays and visualizes forecast results in a Streamlit application.

    Args:
        original_data (pd.DataFrame): Historical macroeconomic data.
        best_chain (list): Order of variable dependencies for forecasting.
        scenario_number (int): Unique identifier for the scenario.
    """
    descriptive_list = [variable_descriptions[code] for code in best_chain]

    selected_descriptions = st.multiselect("Select macroeconomic variables to compare", descriptive_list, key=f"forecast_multiselect{scenario_number}")

    if selected_descriptions:
        min_date = original_data.index.min().date()
        max_date = st.session_state[f'forecasts_{scenario_number}']['FEDFUNDS'].index[-1].date()
        start_date, end_date = st.slider(
            "Select Timeframe for Forecast Comparison",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD",
            key=f"forecast_date_slide{scenario_number}"
        )

        filtered_data = original_data.loc[start_date:end_date]
        filtered_forecasts = {key: value.loc[start_date:end_date] for key, value in st.session_state[f'forecasts_{scenario_number}'].items()}
        fig = plot_selected_forecasts(filtered_data, filtered_forecasts, selected_descriptions, start_date, end_date)
        st.plotly_chart(fig, use_container_width=True, key=f"display_forecast_results_plotly_chart{scenario_number}")

def user_forecast(original_data, best_chain, models, scenario_number=1):
    """
     Manages user interactions to generate and visualize macroeconomic forecasts.

     Args:
         original_data (pd.DataFrame): Historical macroeconomic data.
         best_chain (list): Order of variable dependencies for forecasting.
         models (dict): A dictionary of trained models for forecasting.
         scenario_number (int): Unique identifier for the scenario (default is 1).
     """
    st.header(f"User Forecast (Scenario {scenario_number})")

    # 1. Initialize session state for this specific scenario
    initialize_session_state_scenario(scenario_number, original_data)

    with st.expander("Federal Fund Rate Setting"):
        # 2. Get user inputs specific to this scenario
        st.session_state[f"forecast_months_{scenario_number}"] = st.slider(
            f"Select forecast horizon (months) - Scenario {scenario_number}",
            min_value=1, max_value=24,
            value=st.session_state[f"forecast_months_{scenario_number}"],
            key=f"forecast_horizon_{scenario_number}"
        )
        st.session_state[f"start_value_{scenario_number}"] = st.number_input(
            f"Starting Federal Funds Rate (%) - Scenario {scenario_number}",
            value=st.session_state[f"start_value_{scenario_number}"],
            step=0.25,
            key=f"number_input_{scenario_number}"
        )

        # 3. Adjust Federal Funds Rate
        adjust_fed_funds_rate(scenario_number)

    # 4. Generate and plot the forecast for this scenario
    fed_funds_forecast = create_fedfunds_forecast(
        st.session_state[f"forecast_months_{scenario_number}"],
        st.session_state[f"start_value_{scenario_number}"],
        st.session_state[f"changes_{scenario_number}"]
    )

    start_date = original_data.index[-1] + pd.DateOffset(months=1)
    fig = plot_fedfunds_forecast(fed_funds_forecast, start_date, st.session_state[f"current_month_{scenario_number}"])
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{scenario_number}")

    # 5. Forecast macro variables
    forecast_macros(fed_funds_forecast, original_data, models, best_chain, start_date, scenario_number)

    # Display forecast results for this scenario
    if st.session_state[f"forecasts_{scenario_number}"] is not None:
        display_forecast_results(original_data, best_chain, scenario_number)

def user_forecast_compare(original_data, best_chain, models):
    """
    Allows users to create and compare multiple forecast scenarios in a Streamlit application.

    Args:
        original_data (pd.DataFrame): Historical macroeconomic data.
        best_chain (list): Order of variable dependencies for forecasting.
        models (dict): A dictionary of trained models for forecasting.
    """
    if 'scenario_count' not in st.session_state:
        st.session_state.scenario_count = 1

    st.write("Click 'Add Scenario' to create additional scenarios for analysis. Maximum 3 scenario(s)")

    col1, spacer, col2 = st.columns([1, 0.2, 1])

    with col1:
        if st.session_state.scenario_count < 3:
            if st.button("Add Scenario"):
                st.session_state.scenario_count += 1

    with col2:
        if st.session_state.scenario_count > 1:
            if st.button("Remove Scenario"):
                st.session_state.scenario_count -= 1

    columns = st.columns(st.session_state.scenario_count)
    for i in range(1, st.session_state.scenario_count + 1):
        with columns[i - 1]:
            user_forecast(original_data, best_chain, models, scenario_number=i)
