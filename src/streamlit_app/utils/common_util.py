import pandas as pd
from cryptography.fernet import Fernet

def get_fred_api_key():
    with open("src/resources/keys/fred_api_key.txt", "rb") as file:
        key = file.read()
    cipher_suite = Fernet(key)
    with open("src/resources/keys/fred_api_key_encrypt.txt", "rb") as file:
        encrypted_key = file.read()
    decrypted_key = cipher_suite.decrypt(encrypted_key).decode()
    return decrypted_key

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

def add_recession_highlights(fig):
    recession_dates = get_recession_dates()
    for recession in recession_dates:
        fig.add_vrect(
            x0=recession['from'], x1=recession['to'],
            fillcolor="LightGrey", opacity=0.5,
            layer="below", line_width=0,
        )
    return fig

def load_data(processed_df_filepath: str):
    original_data = pd.read_csv(processed_df_filepath, parse_dates=['date'])
    original_data.set_index('date', inplace=True)

    data_diff = original_data.diff().dropna()
    # print(f'Shape of original data {original_data.shape}')
    # print(f'Date range of original data: {original_data.index.min()} to {original_data.index.max()}')
    # print(f'Shape of differenced data {data_diff.shape}')
    # print(f'Date range of differenced data: {data_diff.index.min()} to {data_diff.index.max()}')
    return original_data, data_diff