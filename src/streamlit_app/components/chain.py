import sys, os
import pandas as pd
from typing import List
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
sys.path.append(os.path.join(os.getcwd(), 'src'))
from streamlit_app.configs.logger_config import logger

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def check_pval_and_vif(model, vif_data, pval_threshold: float, vif_threshold: float) -> bool:
    last_var = model.model.exog_names[-1]
    pvalue = model.pvalues[last_var]

    if pvalue > pval_threshold:
        return False

    max_vif = vif_data['VIF'].max()
    if max_vif > vif_threshold:
        return False

    return True

def evaluate_variable(data: pd.DataFrame, predictive_chain: List[str], column: str, pval_threshold: float) -> pd.DataFrame:
    X = data[predictive_chain]
    X = sm.add_constant(X)
    Y = data[column]

    model = sm.OLS(Y, X).fit()

    vif_data = calculate_vif(X)
    if not check_pval_and_vif(model, vif_data, pval_threshold, vif_threshold=15.0):
        return pd.DataFrame()  # Return an empty DataFrame if conditions are not met

    return pd.DataFrame({
        'Variable': [column],
        'Adj-R-squared': [model.rsquared_adj],
        'Coefficient': [model.params[-1]],  # Last coefficient in params
        'P-value': [model.pvalues[-1]],  # Last p-value in pvalues
        'Max VIF': [vif_data['VIF'].max()]
    })


def find_best_chain(predictive_chain: List[str], data: pd.DataFrame, max_vars: int = None, pval_threshold: float = 0.10):
    results_df = pd.DataFrame(columns=['Variable', 'Adj-R-squared', 'Coefficient', 'P-value', 'Max VIF'])

    if max_vars is None:
        max_vars = len(data.columns) - len(predictive_chain)

    for _ in range(max_vars):
        temp_results_df = pd.DataFrame(columns=['Variable', 'Adj-R-squared', 'Coefficient', 'P-value', 'Max VIF'])
        added = False

        for column in data.columns:
            if column not in predictive_chain:
                variable_result = evaluate_variable(data, predictive_chain, column, pval_threshold)

                if not variable_result.empty:
                    temp_results_df = pd.concat([temp_results_df, variable_result], ignore_index=True)
                    added = True

        if not added:
            logger.info("No variables met the selection criteria.")
            break

        temp_results_df.sort_values(by='Adj-R-squared', inplace=True, ascending=False)
        best_var = temp_results_df.iloc[0]['Variable']
        predictive_chain.append(best_var)

        results_df = pd.concat([results_df, temp_results_df.loc[temp_results_df['Variable'] == best_var]], ignore_index=True)

    logger.info(f"Final predictive chain: {predictive_chain}")
    logger.info(f"Model results:\n{results_df}")

    return predictive_chain