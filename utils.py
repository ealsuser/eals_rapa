from typing import List

import pandas as pd
import numpy as np
import statsmodels.api as sm

def fit_regression(x, y, output_all = False):
    """
    Fits an OLS regression model with intercept to the given predictor and response variables.
    Parameters:
        x (array-like): The predictor variable.
        y (array-like): The response variable.
        output_all (bool, optional): Whether to output all regression results. Defaults to False.
    Returns:
        tuple: A tuple containing the predicted values, lower and upper bounds for RMSE bands, p-value for the slope,
                slope, standard error of the slope, intercept, and standard error of the intercept (if output_all=True).
                Otherwise, returns a tuple containing the predicted values, lower and upper bounds for RMSE bands, and
                p-value for the slope.
    """

    # Add a constant term to the predictor
    x_with_intercept = sm.add_constant(x)

    # Fit the OLS regression model
    model = sm.OLS(y, x_with_intercept).fit()
    y_pred = model.predict(x_with_intercept)

    # Calculate RMSE
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))

    # Get the slope and intercept
    slope = model.params[1]
    slope_se = model.bse[1]
    intercept = model.params[0]
    intercept_se = model.bse[0]

    # Calculate upper and lower bounds for RMSE bands
    upper_bound = y_pred + rmse * 2
    lower_bound = y_pred - rmse * 2

    # Get the slope's p-value
    if len(model.pvalues) > 1:
        p_value = model.pvalues[1]  # The p-value for the slope (the second parameter)
    else:
        p_value = np.nan

    if output_all:
        return y_pred, lower_bound, upper_bound, p_value, slope, slope_se, intercept, intercept_se
    else:
        return y_pred, lower_bound, upper_bound, p_value
