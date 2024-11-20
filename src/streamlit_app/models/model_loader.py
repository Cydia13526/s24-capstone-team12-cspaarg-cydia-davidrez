import pickle
import streamlit as st

class ModelLoader:

    def load_models(variables):
        """
        Loads machine learning models from pickle files.

        This method attempts to load models corresponding to each variable
        provided in the 'variables' list. The models are expected to be
        stored as pickle files named as 'best_model_<variable>.pkl'
        in the 'src/resources/models' directory.

        Args:
            variables (list): A list of strings representing the names of the models to be loaded.

        Returns:
            dict: A dictionary where the keys are the variable names and the values are the loaded models.

        Raises:
            FileNotFoundError: If a model file for a specified variable is not found.
            Exception: If any other error occurs during the loading process.

        Example:
            models = load_models(["sales_forecast", "stock_prediction"])
            # models will contain {'sales_forecast': <model object>, 'stock_prediction': <model object>}
        """
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