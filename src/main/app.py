import os
import streamlit as st
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pipeline import add_engineered_features, load_data, find_best_chain, start_pipeline
from forecaster import Forecaster
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
from datetime import datetime, timedelta
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, HoverTool
from streamlit_bokeh_events import streamlit_bokeh_events
import json
import plotly.express as px
from data_load import get_variable_description, get_all_variable_descriptions, variable_descriptions

# Add this at the beginning of the file, right after the imports
st.set_page_config(layout="wide")

def get_recession_dates():
    recession_dates = [
        {'from': '1953-07-01', 'to': '1954-05-01'},
        {'from': '1957-08-01', 'to': '1958-04-01'},
        {'from': '1960-04-01', 'to': '1961-02-01'},
        {'from': '1969-12-01', 'to': '1970-11-01'},
        {'from': '1973-11-01', 'to': '1975-03-01'},
        {'from': '1980-01-01', 'to': '1980-07-01'},
        {'from': '1981-07-01', 'to': '1982-11-01'},
        {'from': '1990-07-01', 'to': '1991-03-01'},
        {'from': '2001-03-01', 'to': '2001-11-01'},
        {'from': '2007-12-01', 'to': '2009-06-01'},
        {'from': '2020-02-01', 'to': '2020-04-01'}
    ]
    return recession_dates

# Add a new helper function to add recession highlights
def add_recession_highlights(fig):
    recession_dates = get_recession_dates()
    for recession in recession_dates:
        fig.add_vrect(
            x0=recession['from'], x1=recession['to'],
            fillcolor="LightGrey", opacity=0.5,
            layer="below", line_width=0,
        )
    return fig

# Function to load historical data for multiple variables
def load_historical_data(variables):
    db_path = os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")
    conn = sqlite3.connect(db_path)
    if variables:
        query = f"SELECT date, {', '.join(variables)} FROM historical_macro"
    else:
        query = "SELECT * FROM historical_macro"
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()

    return df

def load_prediction_data(variable):
    db_path = os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")
    conn = sqlite3.connect(db_path)
    query = f"SELECT date, {variable}_prediction FROM predictions_macro"
    try:
        df = pd.read_sql_query(query, conn, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {str(e)}")
        st.warning(f"No predictions available for {variable}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        st.error(f"An error occurred while loading prediction data: {str(e)}")
    finally:
        conn.close()
    return pd.DataFrame()

# Function to create a multi-line plot
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

# Function to create a plot with historical and prediction data
def plot_historical_and_prediction(historical_df, prediction_df, variable, variable_description):
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_df.index, 
        y=historical_df[variable], 
        mode='lines', 
        name="Historical"
    ))

    # Plot prediction data if available
    if not prediction_df.empty:
        fig.add_trace(go.Scatter(
            x=prediction_df.index, 
            y=prediction_df[f'{variable}_prediction'], 
            mode='lines', 
            name="Prediction"
        ))

    fig.update_layout(
        title=f'{variable_description}: Historical and Prediction',
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis_tickformat='%Y-%m-%d'
    )

    fig = add_recession_highlights(fig)

    return fig

# Function to load models
def load_models(variables):
    models = {}
    for variable in variables:
        try:
            with open(f"src/resources/models/best_model_{variable}.pkl", 'rb') as file:
                models[variable] = pickle.load(file)
        except FileNotFoundError:
            st.warning(f"Model for {variable} not found.")
        except Exception as e:
            st.error(f"Error loading model for {variable}: {str(e)}")
    return models

# Function to forecast macro variables
def forecast_macro_variables(fedfunds_forecast, original_data, differenced_data, best_chain, models):
    # Convert fedfunds_forecast to DataFrame
    forecast_df = pd.DataFrame({'FEDFUNDS': fedfunds_forecast}, index=fedfunds_forecast.index)
    
    # Calculate the difference for FEDFUNDS
    forecast_df['FEDFUNDS'] = forecast_df['FEDFUNDS'].diff().fillna(forecast_df['FEDFUNDS'].iloc[0] - original_data['FEDFUNDS'].iloc[-1])
    
    # Add engineered features
    forecast_df = add_engineered_features(forecast_df)
    
    # Forecast each variable in the chain
    forecasts = {'FEDFUNDS': fedfunds_forecast}
    for variable in best_chain[1:]:  # Skip FEDFUNDS as it's user-input
        model = models[variable]
        features = Forecaster.fedfund_features()  # Use the static method
        X_forecast = forecast_df[features]
        
        # Make prediction
        y_pred = model.predict(X_forecast)
        
        # Convert back to original scale
        last_actual = original_data[variable].iloc[-1]
        forecasts[variable] = last_actual + np.cumsum(y_pred)
        
        # Add forecast to forecast_df for next variable
        forecast_df[variable] = y_pred
    
    return forecasts

# Function to store forecasts in the database
def store_forecasts(forecasts, original_data):
    db_path = os.path.join(os.getcwd(), "src/resources/data/mongo_db/forecast_macro.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the last date from the original data
    last_date = original_data.index[-1]

    # Create a date range for the forecast period
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(next(iter(forecasts.values()))))

    for var in forecasts:
        if not isinstance(forecasts[var], pd.Series):
            print(f"Warning: forecast for {var} is not a pandas Series. Skipping.")
            continue

        # Ensure the column exists
        cursor.execute(f"PRAGMA table_info(forecast_macro)")
        columns = [row[1] for row in cursor.fetchall()]
        if f"{var}_forecast" not in columns:
            cursor.execute(f"ALTER TABLE forecast_macro ADD COLUMN {var}_forecast REAL")

        for date, value in zip(forecast_dates, forecasts[var]):
            query = f"""
            INSERT OR REPLACE INTO forecast_macro (date, model, {var}_forecast)
            VALUES (?, ?, ?)
            """
            cursor.execute(query, (date.strftime('%Y-%m-%d'), 'user_forecast', float(value)))

    conn.commit()
    conn.close()
    print("Forecasts have been stored in the database.")

# Function to plot forecasts
def plot_forecasts(historical_data, forecasts):
    fig = make_subplots(rows=len(forecasts), cols=1, subplot_titles=list(forecasts.keys()))
    
    for i, (variable, forecast) in enumerate(forecasts.items(), start=1):
        # Plot historical data
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data[variable], name=f"{variable} Historical"),
            row=i, col=1
        )
        
        # Plot forecast
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

def historical_data_comparison(all_variables): ###
    st.header("Historical Data Comparison")
    all_descriptions = get_all_variable_descriptions()
    selected_descriptions = st.multiselect("Select macroeconomic variables to compare", all_descriptions, key="historical_multiselect")
    
    if selected_descriptions:
        selected_variables = [var for var, desc in variable_descriptions.items() if desc in selected_descriptions]
        historical_df = load_historical_data(selected_variables)
        st.write(f"Data range: {historical_df.index.min()} to {historical_df.index.max()}")
        st.write(f"Number of data points: {len(historical_df)}")
        fig = plot_multi_line(historical_df, "Historical Data Comparison", selected_descriptions)
        st.plotly_chart(fig, use_container_width=True)

def load_feature_importances():
    try:
        file_path = os.path.join(os.getcwd(), "src/resources/models/feature_importances.json")
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Feature importances file not found.")
        return {}

def plot_feature_importances(variable_description, importances):
    print(f"Plotting importances for {variable_description}:", importances)
    df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    df['Feature'] = df['Feature'].map(get_variable_description)
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title=f'Top 10 Important Features for {variable_description}')
    fig.update_layout(height=400, width=600)
    return fig

def historical_vs_predictions(all_variables): ###
    st.header("Historical Data vs Predictions")
    all_descriptions = get_all_variable_descriptions()
    selected_description = st.selectbox("Select a macroeconomic variable", all_descriptions, key="prediction_selectbox")
    selected_variable = next(var for var, desc in variable_descriptions.items() if desc == selected_description)
    
    historical_df = load_historical_data([selected_variable])
    prediction_df = load_prediction_data(selected_variable)

    fig = plot_historical_and_prediction(historical_df, prediction_df, selected_variable, selected_description)
    st.plotly_chart(fig)
    
    # Display feature importances
    feature_importances = load_feature_importances()
    if selected_variable in feature_importances:
        st.subheader("Feature Importances")
        fig_importance = plot_feature_importances(selected_description, feature_importances[selected_variable])
        st.plotly_chart(fig_importance)
    else:
        st.info(f"Feature importance data is not available for {selected_description}.")

def plot_selected_forecasts(historical_data, forecasts, selected_variables):
    fig = go.Figure()

    for variable in selected_variables:
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data[variable],
            mode='lines',
            name=f"{variable} Historical"
        ))

        # Plot forecast data
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

def create_fedfunds_forecast(forecast_months, start_value, changes):
    forecast = [start_value]
    for change in changes:
        forecast.append(forecast[-1] + change / 100)  # Convert basis points to percentage
    while len(forecast) < forecast_months + 1:
        forecast.append(forecast[-1])
    return forecast[:forecast_months + 1]

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

    # Set y-axis range
    y_min = min(forecast) - 0.5
    y_max = max(forecast) + 0.5
    fig.update_layout(yaxis_range=[y_min, y_max])

    # Highlight the current forecast month
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

def user_forecast(original_data, best_chain, models): ####
    st.header("User-Defined Federal Funds Rate Forecast")
    
    # Use session state to persist values
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
    
    st.session_state.forecast_months = st.slider("Select forecast horizon (months)", min_value=1, max_value=24, value=st.session_state.forecast_months)
    st.session_state.start_value = st.number_input("Starting Federal Funds Rate (%)", value=st.session_state.start_value, step=0.25)

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
            if st.session_state.current_month >= st.session_state.forecast_months:
                st.session_state.current_month = st.session_state.forecast_months - 1

    if st.button("Reset Forecast"):
        st.session_state.changes = []
        st.session_state.current_month = 0

    fedfunds_forecast = create_fedfunds_forecast(st.session_state.forecast_months, st.session_state.start_value, st.session_state.changes)
    
    start_date = original_data.index[-1] + pd.DateOffset(months=1)
    fig = plot_fedfunds_forecast(fedfunds_forecast, start_date, st.session_state.current_month)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Forecast Macros"):
        fedfunds_forecast_series = pd.Series(fedfunds_forecast[1:], index=pd.date_range(start=start_date, periods=st.session_state.forecast_months, freq='MS'))
        
        print(f"fedfunds_forecast_series: {fedfunds_forecast_series}")
        forecaster = Forecaster(models, original_data, original_data, fedfunds_forecast_series, best_chain, 'FEDFUNDS')
        st.session_state.forecasts = forecaster.forecast(models, original_data, 'FEDFUNDS', 'src/resources/data/mongo_db/', 'models/')
        
        print("Forecasts generated:")
        for var, forecast in st.session_state.forecasts.items():
            print(f"{var}: {forecast}")

        store_forecasts(st.session_state.forecasts, original_data)
    
    if st.session_state.forecasts is not None:
        selected_variable = st.selectbox("Select variable to display", best_chain, index=0)
        
        if selected_variable:
            fig = plot_selected_forecasts(original_data, st.session_state.forecasts, [selected_variable])
            st.plotly_chart(fig, use_container_width=True)

# Update the generate_forecasts function to handle the new forecast format
def generate_forecasts(original_data, best_chain, models, forecast_months, fedfunds_forecast):
    forecast_dates = pd.date_range(start=original_data.index[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    fedfunds_forecast_series = pd.Series(fedfunds_forecast, index=forecast_dates)

    forecaster = Forecaster(models, original_data, fedfunds_forecast_series, best_chain, 'FEDFUNDS')
    forecasts = forecaster.forecast(models, original_data, 'FEDFUNDS', 'data/', 'models/')
    store_forecasts(forecasts, original_data)
    return forecasts

# Main Streamlit app
def main():
    try:
        print("------- Project Set Up Pipeline Started -------")
        # file_path = os.path.join(os.getcwd(), "src/resources/models/feature_importances.json")
        # if os.path.exists(file_path) == False:
        #     original_data, best_chain = start_pipeline()
        #


        print("------- Streamlit App Started -------")
        st.title("Macroeconomic Data and Predictions Dashboard")
        original_data, differenced_data = load_data('src/resources/data/processed/preprocessed_economic_data.csv')

        # Get the best chain from the pipeline
        initial_chain = ['FEDFUNDS']
        best_chain = find_best_chain(initial_chain, differenced_data)
        models = load_models(best_chain[1:])
        all_variables = list(get_all_variable_descriptions())

        tab1, tab2, tab3 = st.tabs(["Historical Data Comparison", "Historical vs Predictions", "User Forecast"])
        with tab1:
            historical_data_comparison(all_variables)
        with tab2:
            historical_vs_predictions(all_variables)
        with tab3:
            user_forecast(original_data, best_chain, models)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()