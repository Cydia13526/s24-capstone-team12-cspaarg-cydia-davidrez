## SIADS Capstone - Federal Funds Rate Scenario and Macroeconomic Prediction App

##### Casey Spaargaren (cspaarg@umich.edu), School of Information, University of Michigan
##### Cydia Tsang (cydia@umich.edu), School of Information, University of Michigan
##### David Rezkalla (davidrez@umich.edu), School of Information, University of Michigan

### Description:
###### This project is a Streamlit-based web application that empowers users to explore how different Federal Funds Rate scenarios could impact macroeconomic factors and the S&P 500. It allows users to define custom Fed Funds Rate scenarios, predict corresponding macroeconomic outcomes, and assess the potential influence on market performance.

### Data Sources:
The data used in this project is accessed through the Federal Reserve Economic Data (FRED) API, which is publicly available and free to use. The API provides economic data and analysis tools maintained by the Federal Reserve Bank of St. Louis. Users can obtain their own FRED API key by visiting the FRED API Key Registration Page and following the instructions for creating a free account.


### Features
#### Scenario Customization:
###### Users can set their own Federal Funds Rate scenarios to simulate real-world monetary policy changes.

#### Predictive Modeling:
###### The app uses predictive models to estimate how macroeconomic indicators (e.g., inflation, unemployment rate) respond to the user-defined scenarios.

#### Market Impact Insights:
###### Forecast how the S&P 500 might react based on the macroeconomic projections.

#### Interactive Dashboard:
###### A clean and intuitive interface to visualize data and predictions dynamically.

### Prerequisites:
###### Python 3.11+

### Installation:
###### 1. git clone https://github.com/Cydia13526/s24-capstone-team12-cspaarg-cydia-davidrez.git
###### 2. git checkout develop
###### 3. pip install -r requirements.txt
###### 4. streamlit run src/streamlit_app/app.py

### Run the Following Command If you failed to import XGBRegressor
###### /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
###### brew install libomp

### File Structure
###### app.py: Main Streamlit application file.
###### /pages: Contains additional pages for the Streamlit multi-page app.
###### /data: Stores any datasets or configuration files.
###### /models: Includes trained models or scripts for prediction.
###### /utils: Utility scripts for data processing or helper functions.
