import pandas as pd
from cryptography.fernet import Fernet

def get_fred_api_key():
    """
    Retrieve and decrypt the FRED API key.

    This function reads the encryption key and the encrypted FRED API key from
    files located in the `src/resources/keys` directory, decrypts the API key,
    and returns it as a string.

    Returns:
        str: The decrypted FRED API key.
    """
    with open("src/resources/keys/fred_api_key.txt", "rb") as file:
        key = file.read()
    cipher_suite = Fernet(key)
    with open("src/resources/keys/fred_api_key_encrypt.txt", "rb") as file:
        encrypted_key = file.read()
    decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
    return decrypted_key

def get_recession_dates():
    """
    Provide a list of historical U.S. recession periods.

    The data includes start and end dates of U.S. recessions from 1953 to 2020.

    Returns:
        list[dict]: A list of dictionaries containing 'from' and 'to' keys
                    representing the start and end dates of recessions.
    """
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

def add_recession_highlights(fig):
    """
    Add recession highlights to a plotly figure.

    This function overlays vertical rectangles representing recession periods
    on a given Plotly figure. The rectangles are semi-transparent and appear
    in the background.

    Args:
        fig (plotly.graph_objects.Figure): The Plotly figure to annotate.

    Returns:
        plotly.graph_objects.Figure: The updated figure with recession highlights.
    """
    recession_dates = get_recession_dates()
    for recession in recession_dates:
        fig.add_vrect(
            x0=recession['from'], x1=recession['to'],
            fillcolor="LightGrey", opacity=0.5,
            layer="below", line_width=0,
        )
    return fig

def load_data(processed_df_filepath: str):
    """
    Load and preprocess time series data.

    This function reads a CSV file containing time series data with a 'date'
    column, sets the 'date' column as the index, and computes the first-order
    differences of the data.

    Args:
        processed_df_filepath (str): The file path to the processed data CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - original_data: The original time series data.
            - data_diff: The differenced time series data, with the first observation removed.
    """
    original_data = pd.read_csv(processed_df_filepath, parse_dates=['date'])
    original_data.set_index('date', inplace=True)

    data_diff = original_data.diff().dropna()
    # print(f'Shape of original data {original_data.shape}')
    # print(f'Date range of original data: {original_data.index.min()} to {original_data.index.max()}')
    # print(f'Shape of differenced data {data_diff.shape}')
    # print(f'Date range of differenced data: {data_diff.index.min()} to {data_diff.index.max()}')
    return original_data, data_diff