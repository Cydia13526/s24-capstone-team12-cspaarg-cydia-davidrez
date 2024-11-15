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

def plot_selected_forecasts(historical_df, forecast_df, selected_variables):
    fig = go.Figure()

    for variable in selected_variables:
        fig.add_trace(go.Scatter(
            x=historical_df.index,
            y=historical_df[variable],
            mode='lines',
            name=f"{variable} Historical"
        ))

        forecast_dates = pd.date_range(start=historical_df.index[-1] + pd.DateOffset(months=1), periods=len(forecast_df[variable]), freq='MS')
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_df[variable],
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

def plot_multi_line(df, title, variable_descriptions):
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


def initialize_session_state_scenario(scenario_number, original_data):
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
    col1, col2, col3 = st.columns(3)

    with col1:
        change = st.selectbox("Change", ["-50 bps", "-25 bps", "No change", "+25 bps", "+50 bps"], index=2, key=f"fed_fund_rate_select_box{scenario_number}")
    with col2:
        months = st.number_input("Apply for how many months?", min_value=1, max_value=st.session_state[f'forecast_months_{scenario_number}'] - st.session_state[f'current_month_{scenario_number}'] , value=1, key=f"fed_fund_rate_number_input{scenario_number}")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Apply Change", key=f"apply_change_{scenario_number}"):
            change_value = int(change.split()[0]) if change != "No change" else 0
            st.session_state[f'changes_{scenario_number}'].extend([change_value] * months)
            st.session_state[f'current_month_{scenario_number}'] += months
            if st.session_state[f'current_month_{scenario_number}']  >= st.session_state[f'forecast_months_{scenario_number}']:
                st.session_state[f'current_month_{scenario_number}']  = st.session_state[f'forecast_months_{scenario_number}'] - 1


def forecast_macros(fed_funds_forecast, original_data, models, best_chain, start_date, scenario_number):
    fed_funds_forecast_series = pd.Series(fed_funds_forecast[1:], index=pd.date_range(start=start_date, periods=st.session_state[f'forecast_months_{scenario_number}'], freq='MS'))

    logger.info(f"fed_funds_forecast_series: {fed_funds_forecast_series}")
    forecaster = Forecaster(models, original_data, original_data, fed_funds_forecast_series, best_chain, 'FEDFUNDS')
    st.session_state[f'forecasts_{scenario_number}'] = forecaster.forecast(models, original_data, 'FEDFUNDS', 'src/resources/data/mongo_db/', 'models/')

    logger.info("Forecasts generated:")
    for var, forecast in st.session_state[f'forecasts_{scenario_number}'].items():
        logger.info(f"{var}: {forecast}")

    store_forecasts(st.session_state[f'forecasts_{scenario_number}'], original_data)

def display_forecast_results(original_data, best_chain, scenario_number):
    # selected_variable = st.selectbox("Select variable to display", best_chain, index=0)
    # all_descriptions = get_all_variable_descriptions()
    selected_descriptions = st.multiselect("Select macroeconomic variables to compare", best_chain, key=f"forecast_multiselect{scenario_number}")

    if selected_descriptions:
        fig = plot_selected_forecasts(original_data, st.session_state[f'forecasts_{scenario_number}'], selected_descriptions)
        st.plotly_chart(fig, use_container_width=True, key=f"display_forecast_results_plotly_chart{scenario_number}")

def add_scenario():
    st.session_state.scenario_count += 1

def user_forecast(original_data, best_chain, models, scenario_number=1):
    st.header(f"User-Defined Federal Funds Rate Forecast - Scenario {scenario_number}")

    # 1. Initialize session state for this specific scenario
    initialize_session_state_scenario(scenario_number, original_data)

    # 2. Reset forecast if requested for this scenario
    if st.button(f"Reset Forecast - Scenario {scenario_number}", key=f"reset_forecast_{scenario_number}"):
        st.session_state[f"changes_{scenario_number}"] = []
        st.session_state[f"current_month_{scenario_number}"] = 0

    # 3. Get user inputs specific to this scenario
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

    # 4. Adjust Federal Funds Rate
    adjust_fed_funds_rate(scenario_number)


    # 5. Generate and plot the forecast for this scenario
    fed_funds_forecast = create_fedfunds_forecast(
        st.session_state[f"forecast_months_{scenario_number}"],
        st.session_state[f"start_value_{scenario_number}"],
        st.session_state[f"changes_{scenario_number}"]
    )

    start_date = original_data.index[-1] + pd.DateOffset(months=1)
    fig = plot_fedfunds_forecast(fed_funds_forecast, start_date, st.session_state[f"current_month_{scenario_number}"])
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{scenario_number}")

    # 6. Forecast macro variables
    forecast_macros(fed_funds_forecast, original_data, models, best_chain, start_date, scenario_number)

    # Display forecast results for this scenario
    if st.session_state[f"forecasts_{scenario_number}"] is not None:
        display_forecast_results(original_data, best_chain, scenario_number)

def user_forecast_compare(original_data, best_chain, models):
    if 'scenario_count' not in st.session_state:
        st.session_state.scenario_count = 1

    st.write("Click 'Add Scenario' to create additional scenarios for analysis. (Maximum 3 scenarios)")

    col1, col2 = st.columns(2)
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
