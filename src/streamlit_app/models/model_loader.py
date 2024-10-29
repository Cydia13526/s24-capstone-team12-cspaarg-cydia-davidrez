import pickle
import streamlit as st

class ModelLoader:

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