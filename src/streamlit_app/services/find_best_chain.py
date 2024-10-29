import pandas as pd
from typing import List
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def find_best_chain(predictive_chain: List[str], data: pd.DataFrame, max_vars: int = None, pval_threshold: float = 0.10):
    results_df = pd.DataFrame(columns=['Variable', 'Adj-R-squared', 'Coefficient', 'P-value', 'Max VIF'])

    if max_vars is None:
        max_vars = len(data.columns) - len(predictive_chain)

    for _ in range(max_vars):
        temp_results_df = pd.DataFrame(columns=['Variable', 'Adj-R-squared', 'Coefficient', 'P-value', 'Max VIF'])
        added = False

        for column in data.columns:
            if column not in predictive_chain:
                X = data[predictive_chain]
                X = sm.add_constant(X)
                Y = data[column]

                model = sm.OLS(Y, X).fit()
                adj_rsquared = model.rsquared_adj
                coefficients = model.params
                pvalues = model.pvalues

                # Access p-value of the last predictor
                last_var = predictive_chain[-1]
                if pvalues[last_var] > pval_threshold:
                    continue

                vif_data = pd.DataFrame()
                vif_data['feature'] = X.columns
                vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

                max_vif = vif_data['VIF'].max()

                vif_threshold = 15.0
                if max_vif > vif_threshold:
                    continue

                current_result = pd.DataFrame({
                    'Variable': [column],
                    'Adj-R-squared': [adj_rsquared],
                    'Coefficient': [coefficients[last_var]],
                    'P-value': [pvalues[last_var]],
                    'Max VIF': [max_vif]
                })
                temp_results_df = pd.concat([temp_results_df, current_result], ignore_index=True)
                added = True

        if not added:
            print("No variables met the selection criteria.")
            break

        # Select the best variable based on Adjusted R-squared
        temp_results_df.sort_values(by='Adj-R-squared', inplace=True, ascending=False)
        best_var = temp_results_df.iloc[0]['Variable']
        predictive_chain.append(best_var)

        # Update results
        results_df = pd.concat([results_df, temp_results_df.loc[temp_results_df['Variable'] == best_var]], ignore_index=True)

    print("Final predictive chain:", predictive_chain)
    print("Model results:\n", results_df)

    return predictive_chain