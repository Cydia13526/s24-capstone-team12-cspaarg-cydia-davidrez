import sys, os
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.data.fred_data_loader import get_all_variable_descriptions, variable_descriptions, get_variable_description
from streamlit_app.utils.common_util import add_recession_highlights
from streamlit_app.data.mongo_data_loader import MongoDataLoader

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

def historical_data_comparison(all_variables):
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