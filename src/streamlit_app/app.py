import os, sys
import streamlit as st
from pipeline import start_pipeline
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.pages.historical_data_comparison_tab import historical_data_comparison
from streamlit_app.data.fred_data_loader import get_all_variable_descriptions, variable_descriptions, get_variable_description
from streamlit_app.pages.historical_vs_predictions_tab import historical_vs_predictions
from streamlit_app.pages.user_forecast_tab import user_forecast_compare
from streamlit_app.models.model_loader import ModelLoader
from streamlit_app.components.chain import find_best_chain
from streamlit_app.utils.common_util import load_data
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
# Add this at the beginning of the file, right after the imports

st.set_page_config(
    page_title="Macroeconomic Data and Predictions Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={}
)

# Main Streamlit app
def main():
    """
    This function is the main entry point for the Streamlit app. It initializes the dashboard by loading the necessary
    data, models, and components, then displays multiple tabs for comparison and forecasting of macroeconomic data.

    The flow of the app is as follows:
    1. Check if the feature importances file exists to decide whether to run the pipeline or load preprocessed data.
    2. Based on the data, select the best chain of variables for model prediction.
    3. Set up the Streamlit page configuration and title.
    4. Load the models based on the selected best chain.
    5. Create three tabs for different functionalities: historical data comparison, historical vs predictions, and user forecast.
    6. Display content within each tab using predefined functions for each task.

    If an error occurs during execution, an error message is displayed.

    Returns:
        None: This function operates by side effects (streamlit app rendering).
    """
    try:
        file_path = os.path.join(os.getcwd(), "src/resources/models/feature_importances.json")
        if os.path.exists(file_path) == False:
            original_data, best_chain = start_pipeline()
        else:
            original_data, differenced_data = load_data('src/resources/data/processed/preprocessed_economic_data.csv')

            initial_chain = ['FEDFUNDS']
            best_chain = find_best_chain(initial_chain, differenced_data)

        st.title("Macroeconomic Data and Predictions Dashboard")

        models = ModelLoader.load_models(best_chain[1:])
        all_variables = list(get_all_variable_descriptions())

        tab1, tab2, tab3 = st.tabs(["Historical Data Comparison", "Historical vs Predictions", "User Forecast"])
        with tab1:
            historical_data_comparison(all_variables)
        with tab2:
            historical_vs_predictions(all_variables)
        with tab3:
            user_forecast_compare(original_data, best_chain, models)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()