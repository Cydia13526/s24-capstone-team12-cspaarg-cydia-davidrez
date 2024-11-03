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

def plot_fedfunds_forecast(forecast, start_date, current_month):
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

def plot_selected_forecasts(historical_data, forecasts, selected_variables):
    fig = go.Figure()

    for variable in selected_variables:
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[variable],
            mode='lines',
            name=f"{variable} Historical"
        ))

        forecast_dates = pd.date_range(start=historical_data.index[-1] + pd.DateOffset(months=1), periods=len(forecasts[variable]), freq='MS')
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecasts[variable],
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
        )
    )

    fig = add_recession_highlights(fig)  # Add this line

    return fig

def plot_forecasts(historical_data, forecasts):
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

def create_fedfunds_forecast(forecast_months, start_value, changes):
    forecast = [start_value]
    for change in changes:
        forecast.append(forecast[-1] + change / 100)  # Convert basis points to percentage
    while len(forecast) < forecast_months + 1:
        forecast.append(forecast[-1])
    return forecast[:forecast_months + 1]

def forecast_macro_variables(fedfunds_forecast, original_data, differenced_data, best_chain, models):
    forecast_df = pd.DataFrame({'FEDFUNDS': fedfunds_forecast}, index=fedfunds_forecast.index)
    forecast_df['FEDFUNDS'] = forecast_df['FEDFUNDS'].diff().fillna(forecast_df['FEDFUNDS'].iloc[0] - original_data['FEDFUNDS'].iloc[-1])
    diff_df = forecast_df.copy()

    forecast_df = Forecaster.add_engineered_features(diff_df)
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
    forecast_dates = pd.date_range(start=original_data.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    fed_funds_forecast_series = pd.Series(fedfunds_forecast, index=forecast_dates)

    forecaster = Forecaster(models, original_data, fed_funds_forecast_series, best_chain, 'FEDFUNDS')
    forecasts = forecaster.forecast(models, original_data, 'FEDFUNDS', 'data/', 'models/')
    store_forecasts(forecasts, original_data)
    return forecasts


def initialize_session_state(original_data):
    if 'forecast_months' not in st.session_state:
        st.session_state.forecast_months = 12
    if 'start_value' not in st.session_state:
        st.session_state.start_value = original_data['FEDFUNDS'].iloc[-1]
    if 'changes' not in st.session_state:
        st.session_state.changes = []
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = None
    if 'current_month' not in st.session_state:
        st.session_state.current_month = 0

def adjust_fed_funds_rate():
    st.subheader("Adjust Federal Funds Rate")
    col1, col2, col3 = st.columns(3)

    with col1:
        change = st.selectbox("Change", ["-50 bps", "-25 bps", "No change", "+25 bps", "+50 bps"], index=2)
    with col2:
        months = st.number_input("Apply for how many months?", min_value=1, max_value=st.session_state.forecast_months - st.session_state.current_month, value=1)
    with col3:
        if st.button("Apply Change"):
            change_value = int(change.split()[0]) if change != "No change" else 0
            st.session_state.changes.extend([change_value] * months)
            st.session_state.current_month += months
            st.session_state.current_month = min(st.session_state.current_month, st.session_state.forecast_months - 1)

def forecast_macros(fed_funds_forecast, original_data, models, best_chain, start_date):
    fed_funds_forecast_series = pd.Series(fed_funds_forecast[1:], index=pd.date_range(start=start_date, periods=st.session_state.forecast_months, freq='MS'))

    logger.info(f"fed_funds_forecast_series: {fed_funds_forecast_series}")
    forecaster = Forecaster(models, original_data, original_data, fed_funds_forecast_series, best_chain, 'FEDFUNDS')
    st.session_state.forecasts = forecaster.forecast(models, original_data, 'FEDFUNDS', 'src/resources/data/mongo_db/', 'models/')

    logger.info("Forecasts generated:")
    for var, forecast in st.session_state.forecasts.items():
        logger.info(f"{var}: {forecast}")

    store_forecasts(st.session_state.forecasts, original_data)

def display_forecast_results(original_data, best_chain):
    if st.session_state.forecasts is not None:
        selected_variable = st.selectbox("Select variable to display", best_chain, index=0)

        if selected_variable:
            fig = plot_selected_forecasts(original_data, st.session_state.forecasts, [selected_variable])
            st.plotly_chart(fig, use_container_width=True)


def user_forecast(original_data, best_chain, models):
    st.header("User-Defined Federal Funds Rate Forecast")

    # 1. Initialize session state
    initialize_session_state(original_data)

    # 2. Get user inputs
    st.session_state.forecast_months = st.slider("Select forecast horizon (months)", min_value=1, max_value=24, value=st.session_state.forecast_months)
    st.session_state.start_value = st.number_input("Starting Federal Funds Rate (%)", value=st.session_state.start_value, step=0.25)

    # 3. Adjust Federal Funds Rate
    adjust_fed_funds_rate()

    # 4. Reset forecast if requested
    if st.button("Reset Forecast"):
        st.session_state.changes = []
        st.session_state.current_month = 0

    # 5. Generate and plot the forecast
    fed_funds_forecast = create_fedfunds_forecast(st.session_state.forecast_months, st.session_state.start_value, st.session_state.changes)

    start_date = original_data.index[-1] + pd.DateOffset(months=1)
    fig = plot_fedfunds_forecast(fed_funds_forecast, start_date, st.session_state.current_month)
    st.plotly_chart(fig, use_container_width=True)

    # 6. Forecast macro variables
    if st.button("Forecast Macros"):
        forecast_macros(fed_funds_forecast, original_data, models, best_chain, start_date)

    # Display forecast results
    display_forecast_results(original_data, best_chain)