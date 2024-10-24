import pandas as pd
from fredapi import fred
from dotenv import load_dotenv
from datetime import datetime as dt
from common_util import get_fred_api_key

load_dotenv()

variable_descriptions = {
    "MORTGAGE30US": "30-Year Mortgage Rate",
    "HOUST": "New Housing Units Started",
    "GDPC1": "Real Gross Domestic Product",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Rate",
    "DSPIC96": "Real Disposable Personal Income",
    "PSAVERT": "Personal Saving Rate",
    "UMCSENT": "Consumer Sentiment",
    "CSUSHPISA": "National Home Price Index",
    "SP500": "S&P 500",
    "DRTSCILM": "Consumer Loan Delinquency Rate",
    "CES0500000003": "Average Hourly Earnings",
    "HDTGPDUSQ163N": "Household Debt to GDP",
    "CP": "Corporate Profits After Tax",
    "INDPRO": "Industrial Production Index",
    "BOPGSTB": "Trade Balance",
    "DGS2": "2-Year Treasury Rate",
    "T10Y2Y": "10-Year Treasury Minus 2-Year Treasury"
}

# Function to get the description for a given variable code
def get_variable_description(variable_code):
    return variable_descriptions.get(variable_code, variable_code)

# Add this new function to get all variable descriptions
def get_all_variable_descriptions():
    return list(variable_descriptions.values())



class FREDDataLoader:
    def __init__(self):
        self.fred = fred.Fred(api_key=get_fred_api_key())
        self.variables = [
                "MORTGAGE30US", # 30-Year mortgage rate
                "HOUST",        # New Privately-Owned Housing Units Started
                "GDPC1",        # Real Gross Domestic Product
                "UNRATE",       # Unemployment Rate
                "CPIAUCSL",     # Consumer Price Index for All Urban Consumers: All Items
                "FEDFUNDS",     # Federal Funds Rate
                "DGS10",        # 10-Year Treasury Constant Maturity Rate
                "DSPIC96",      # Real Disposable Personal Income
                "PSAVERT",      # Personal Saving Rate
                "UMCSENT",      # University of Michigan: Consumer Sentiment
                "CSUSHPISA",    # S&P/Case-Shiller U.S. National Home Price Index
                "SP500",        # S&P 500
                "DRTSCILM",     # Delinquency Rate on Consumer Loans, All Commercial Banks
                "CES0500000003",# Average Hourly Earnings of All Employees, Total Private
                "HDTGPDUSQ163N",# Household Debt to GDP for United States
                "CP",           # Corporate Profits After Tax
                "INDPRO",       # Industrial Production Index
                "BOPGSTB",      # Trade Balance: Goods and Services, Balance of Payments Basis
                "DGS2",         # 2-Year Treasury Constant Maturity Rate
                "T10Y2Y"        # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
            ]
        

    def load_data(self, start_date='1954-07-01', end_date=dt.today().strftime('%Y-%m-%d')):
        data = {}
        for var in self.variables:
            series = self.fred.get_series(var, observation_start=start_date, observation_end=end_date)
            data[var] = series

        df = pd.DataFrame(data)
        return df
    
    def preprocess_data(self, df):
        preprocessed_df = df.copy()
        preprocessed_df.fillna(method='ffill', inplace=True)
        preprocessed_df.fillna(method='bfill', inplace=True)
        preprocessed_df.index.names = ['date']

        preprocessed_df = preprocessed_df.resample('MS').first()

        return preprocessed_df



    def save_data(self, df, filename='preprocessed_economic_data.csv'):
        df.to_csv(f'../data/processed/{filename}', index=True)
        print(f'Data saved to data/processed/{filename}')

    def get_recession_dates(self):
        return self.recession_dates
    

if __name__ == '__main__':
    loader = FREDDataLoader()
    df = loader.load_data()
    preprocessed_df = loader.preprocess_data(df)
    loader.save_data(preprocessed_df)

