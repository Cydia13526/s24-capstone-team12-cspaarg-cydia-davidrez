"""
This script defines a Streamlit app for visualizing macroeconomic variables and their predictions.

Key functionalities include:
- Plotting historical and predicted data for a selected macroeconomic variable.
- Displaying feature importances for the selected variable.
- Loading data from MongoDB databases and a JSON file.

"""
import json
import sys, os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.mongo_data_loader import MongoDataLoader
from streamlit_app.utils.common_util import add_recession_highlights
from streamlit_app.data.fred_data_loader import get_all_variable_descriptions, variable_descriptions, get_variable_description
from streamlit_app.configs.logger_config import logger

def plot_historical_and_prediction(historical_df, prediction_df, variable, variable_description):
    """
    Creates a Plotly figure showing historical data and predictions for a macroeconomic variable.

    Args:
        historical_df (pd.DataFrame): DataFrame containing historical data.
        prediction_df (pd.DataFrame): DataFrame containing prediction data.
        variable (str): The name of the variable to plot.
        variable_description (str): A human-readable description of the variable.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df[variable],
        mode='lines',
        name="Historical"
    ))

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

def plot_feature_importances(variable_description, importances):
    """
    Generates a bar chart showing the top feature importances for a macroeconomic variable.

    Args:
        variable_description (str): Human-readable description of the variable.
        importances (dict): Dictionary of feature names and their importances.

    Returns:
        px.bar: A Plotly bar chart figure object.
    """
    logger.info(f"Plotting importances for {variable_description}:", importances)
    df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    df['Feature'] = df['Feature'].map(get_variable_description)
    df = df.sort_values('Importance', ascending=True)

    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title=f'Top 10 Important Features for {variable_description}')
    fig.update_layout(height=400, width=600)
    return fig

def load_feature_importances():
    """
    Loads feature importances from a JSON file.

    Returns:
        dict: A dictionary mapping variables to their feature importance data.
              Returns an empty dictionary if the file is not found.
    """
    try:
        file_path = os.path.join(os.getcwd(), "src/resources/models/feature_importances.json")
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Feature importances file not found.")
        return {}

def historical_vs_predictions(all_variables):
    """
    Streamlit UI component for comparing historical data and predictions
    of macroeconomic variables.

    Args:
        all_variables (list): List of available macroeconomic variables.

    Streamlit Components:
        - Header for the comparison section.
        - Dropdown to select a macroeconomic variable.
        - Plotly charts for historical data vs predictions and feature importances.
    """
    st.header("Historical Data vs Predictions")
    all_descriptions = get_all_variable_descriptions()
    selected_description = st.selectbox("Select a macroeconomic variable", all_descriptions, key="prediction_selectbox")
    selected_variable = next(var for var, desc in variable_descriptions.items() if desc == selected_description)

    with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")) as loader:
        historical_df = loader.load_historical_data([selected_variable])

    with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")) as loader:
        prediction_df = loader.load_prediction_data([selected_variable])

    fig = plot_historical_and_prediction(historical_df, prediction_df, selected_variable, selected_description)
    st.plotly_chart(fig)

    feature_importances = load_feature_importances()
    if selected_variable in feature_importances:
        st.subheader("Feature Importances")
        fig_importance = plot_feature_importances(selected_description, feature_importances[selected_variable])
        st.plotly_chart(fig_importance)
    else:
        st.info(f"Feature importance data is not available for {selected_description}.")