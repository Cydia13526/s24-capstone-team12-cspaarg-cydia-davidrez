"""
This module provides functions for visualizing historical macroeconomic data
and comparing multiple variables using Streamlit and Plotly.

Functions:
    plot_multi_line(df, title, variable_descriptions):
        Generates a multi-line Plotly figure to visualize data with highlighted recession periods.

    historical_data_comparison(all_variables):
        Creates a Streamlit interface for selecting and comparing historical macroeconomic variables.

"""

import sys, os
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.fred_data_loader import get_all_variable_descriptions, variable_descriptions, get_variable_description
from streamlit_app.utils.common_util import add_recession_highlights
from streamlit_app.data.mongo_data_loader import MongoDataLoader

def plot_multi_line(df, title, variable_descriptions):
    """
    Generates a multi-line Plotly figure to visualize a DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame where rows represent time-series data and columns are variables.
        title (str): Title for the figure.
        variable_descriptions (list of str): Descriptions corresponding to the DataFrame columns.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with multi-line plots and recession highlights.
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

def historical_data_comparison(all_variables):
    """
    Streamlit interface for historical data comparison.

    Displays a multi-select widget for choosing macroeconomic variables to compare.
    Loads the selected variables' historical data from a MongoDB database.
    Visualizes the data with a multi-line plot including a range slider and recession highlights.

    Args:
      all_variables (list of str): A list of all available macroeconomic variables.

    Returns:
      None
    """
    st.header("Historical Data Comparison")
    all_descriptions = get_all_variable_descriptions()
    selected_descriptions = st.multiselect("Select macroeconomic variables to compare", all_descriptions, key="historical_multiselect")

    if selected_descriptions:
        selected_variables = [var for var, desc in variable_descriptions.items() if desc in selected_descriptions]
        with MongoDataLoader(os.path.join(os.getcwd(), "src/resources/data/mongo_db/historical_macro.db")) as loader:
            historical_df = loader.load_historical_data(selected_variables)
        st.write(f"Data range: {historical_df.index.min()} to {historical_df.index.max()}")
        st.write(f"Number of data points: {len(historical_df)}")
        fig = plot_multi_line(historical_df, "Historical Data Comparison", selected_descriptions)
        st.plotly_chart(fig, use_container_width=True)