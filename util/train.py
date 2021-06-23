import numpy as np
import pandas as pd
import util.dictionaries as di


############################################### Train Model Helpers #################################################


def fit_models(algo, x_df, y_df):
    """
    Takes an algorithm, X, y and fits it and returns the model.
    """
    for i in range(len(algo)):
        algo[i] = algo[i].fit(x_df, y_df)
        print('done fit ' + str(algo[i])[:10])
    return list(algo)

