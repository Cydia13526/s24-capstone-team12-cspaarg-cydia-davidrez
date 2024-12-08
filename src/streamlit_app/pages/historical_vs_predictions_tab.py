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
import numpy as np
import scipy.stats as stats
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

def load_model_metrics():
    """
    Loads model performance metrics from JSON file.

    Returns:
        dict: Dictionary containing model metrics for each variable
    """
    try:
        file_path = os.path.join(os.getcwd(), "src/resources/models/model_metrics.json")
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Model metrics file not found.")
        return {}

def display_metrics(metrics):
    """
    Displays model performance metrics in a formatted way.
    """
    st.subheader("Model Performance Metrics")
    
    # First show original scale metrics as they're more intuitive
    st.markdown("##### Metrics on Original Scale")
    st.markdown("*These metrics show how well the model performs on the actual values you see in the chart above.*")
    metrics_df = pd.DataFrame({
        'Metric': ['Root Mean Square Error (RMSE)', 
                  'Mean Absolute Error (MAE)',
                  'R-squared (R²)',
                  'Mean Absolute Percentage Error (MAPE)'],
        'Value': [
            f"{metrics['actual']['RMSE']:.4f}",
            f"{metrics['actual']['MAE']:.4f}",
            f"{metrics['actual']['R2']:.4f}",
            f"{metrics['actual']['MAPE']:.2f}%"
        ]
    })
    st.dataframe(metrics_df, hide_index=True)
    
    # Add expander for technical metrics
    with st.expander("Show Technical Metrics (Differenced Scale)"):
        st.markdown("*These metrics show model performance on the differenced data used during training.*")
        metrics_df = pd.DataFrame({
            'Metric': ['Root Mean Square Error (RMSE)', 
                      'Mean Absolute Error (MAE)',
                      'R-squared (R²)',
                      'Mean Absolute Percentage Error (MAPE)'],
            'Value': [
                f"{metrics['differenced']['RMSE']:.4f}",
                f"{metrics['differenced']['MAE']:.4f}",
                f"{metrics['differenced']['R2']:.4f}",
                f"{metrics['differenced']['MAPE']:.2f}%"
            ]
        })
        st.dataframe(metrics_df, hide_index=True)

def plot_residuals(y_true, y_pred, dates):
    """
    Creates a residual plot showing prediction errors over time.
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode='lines',
        name='Residuals'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title='Residual Plot (Prediction Errors Over Time)',
        xaxis_title='Date',
        yaxis_title='Residual (Actual - Predicted)',
        height=400
    )
    return fig

def plot_qq(residuals):
    fig = go.Figure()
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Fit'))
    fig.update_layout(
        title='Q-Q Plot',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Ordered Values',
        height=400
    )
    return fig

def plot_mae_over_time(mae_over_time, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=mae_over_time,
        mode='lines',
        name='MAE Over Time'
    ))
    fig.update_layout(
        title='MAE Over Time',
        xaxis_title='Date',
        yaxis_title='MAE',
        height=400
    )
    return fig

def plot_correlation(historical_df, variable, variable_description):
    """
    Creates a scatter plot showing correlation between FEDFUNDS and the selected variable.
    """
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=historical_df['FEDFUNDS'],
        y=historical_df[variable],
        mode='markers',
        name='Data Points',
        marker=dict(
            size=8,
            opacity=0.6
        )
    ))
    
    coefficients = np.polyfit(historical_df['FEDFUNDS'], historical_df[variable], 1)
    correlation = np.corrcoef(historical_df['FEDFUNDS'], historical_df[variable])[0,1]
    
    x_range = np.array([historical_df['FEDFUNDS'].min(), historical_df['FEDFUNDS'].max()])
    y_range = coefficients[0] * x_range + coefficients[1]
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name=f'Correlation Line (r={correlation:.3f})',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Correlation: Federal Funds Rate vs {variable_description}',
        xaxis_title='Federal Funds Rate',
        yaxis_title=variable_description,
        height=400
    )
    
    return fig

def historical_vs_predictions(all_variables):
    """
    Streamlit UI component for comparing historical data and predictions
    of macroeconomic variables.
    """
    st.header("Historical Data vs Predictions")
    all_descriptions = get_all_variable_descriptions()
    selected_description = st.selectbox("Select a macroeconomic variable", all_descriptions, key="prediction_selectbox")
    selected_variable = next(var for var, desc in variable_descriptions.items() if desc == selected_description)

    with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")) as loader:
        variables_to_load = [selected_variable, 'FEDFUNDS']
        historical_df = loader.load_historical_data(variables_to_load)
        
        # st.write("Debug Information:")
        # st.write("Historical DataFrame Columns:", historical_df.columns.tolist())
        # st.write("Historical DataFrame Shape:", historical_df.shape)

    with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/predictions_macro.db")) as loader:
        prediction_df = loader.load_prediction_data([selected_variable])

    # Main prediction plot
    fig = plot_historical_and_prediction(historical_df, prediction_df, selected_variable, selected_description)
    st.plotly_chart(fig)

    # Display metrics if available
    model_metrics = load_model_metrics()
    if selected_variable in model_metrics:
        display_metrics(model_metrics[selected_variable])
    else:
        st.info(f"Performance metrics are not available for {selected_description}.")

    # Display feature importances 
    feature_importances = load_feature_importances()
    if selected_variable in feature_importances:
        st.subheader("Feature Importances")
        fig_importance = plot_feature_importances(selected_description, feature_importances[selected_variable])
        st.plotly_chart(fig_importance)
    else:
        st.info(f"Feature importance data is not available for {selected_description}.")

    if not prediction_df.empty:
        try:
            prediction_df.index = pd.to_datetime(prediction_df.index)
            historical_df.index = pd.to_datetime(historical_df.index)
            
            pred_col = f'{selected_variable}_prediction'
            common_dates = historical_df.index.intersection(prediction_df.index)
            
            if len(common_dates) > 0:
                historical_values = historical_df.loc[common_dates, selected_variable]
                predicted_values = prediction_df.loc[common_dates, pred_col]
                
                residuals = historical_values - predicted_values
                
                st.subheader("Model Performance Visualizations")
                
                fig_residuals = plot_residuals(
                    historical_values,
                    predicted_values,
                    common_dates
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Q-Q plot
                    fig_qq = plot_qq(residuals)
                    st.plotly_chart(fig_qq, use_container_width=True)
                
                with col2:
                    
                    # Correlation plot
                    fig_corr = plot_correlation(historical_df, selected_variable, selected_description)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
            else:
                st.warning("No overlapping dates found between historical and prediction data.")
                
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            logger.error(f"Visualization error: {str(e)}", exc_info=True)