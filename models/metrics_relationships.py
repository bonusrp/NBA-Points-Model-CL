import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from util import bet
import util.dictionaries as di

import os
from pathlib import Path

from tqdm import tqdm
from functools import partial

from sklearn import preprocessing, feature_selection
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb

import optuna


def objective(trial, train_x, train_y, train_bet, val_x, val_y, val_bet, test_x, test_y, test_bet, relationship):
    # parameter space to test
    param = {
        "objective": 'reg:pseudohubererror',
        "tree_method": "hist",
        "booster": "gbtree",
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        # L1 regularization weight.
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.20, 0.8),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": 0.5,
        "n_jobs": 6,
        "n_estimators": 50,
        "random_state": 12,
        "verbosity": 0,
        "base_score": 100
    }

    # Since XGBRegressor doesnt take a param argument rather just all the params as individual named arguments
    # use **kargs
    model_xgb = xgb.XGBRegressor(**param).fit(train_x, train_y)
    pred_val = model_xgb.predict(val_x)
    pred_test = model_xgb.predict(test_x)

    # Getting mae objective for both val set and test set to see if they correlate
    mae_val = metrics.mean_absolute_error(val_y, pred_val)
    mae_test = metrics.mean_absolute_error(test_y, pred_test)

    # Getting mae objective for both val set and test set to see if they correlate
    exvar_val = metrics.explained_variance_score(val_y, pred_val)
    exvar_test = metrics.explained_variance_score(test_y, pred_test)

    # Getting mae objective for both val set and test set to see if they correlate
    med_val = metrics.median_absolute_error(val_y, pred_val)
    med_test = metrics.median_absolute_error(test_y, pred_test)

    # Getting mae objective for both val set and test set to see if they correlate
    rmse_val = metrics.mean_squared_error(val_y, pred_val, squared=False)
    rmse_test = metrics.mean_squared_error(test_y, pred_test, squared=False)

    # Getting mae objective for both val set and test set to see if they correlate
    pred_var_val = pred_val.var()
    pred_var_test = pred_test.var()

    # Getting dollars won objective for both val set and test set to see if they correlate
    ml_outcome_val, ml_win_pct_val, ml_dollars_val = \
        bet.ml_outcome(pred_val, val_bet.loc[:, ['ml_home', 'ml_away']],
                       val_bet.home_win, margin=0, lower_bound_odds=0)

    spread_outcome_val, spread_win_pct_val, spread_dollars_val = \
        bet.spread_outcome(pred_val, val_bet.spread, val_bet.home_margin,
                           margin=1)

    ml_outcome_test, ml_win_pct_test, ml_dollars_test = \
        bet.ml_outcome(pred_test, test_bet.loc[:, ['ml_home', 'ml_away']],
                       test_bet.home_win, margin=0, lower_bound_odds=0)

    if relationship == 'mae':
        return mae_val, ml_dollars_test
    elif relationship == 'med':
        return med_val, ml_dollars_test
    elif relationship == 'ex_var':
        return exvar_val, ml_dollars_test
    elif relationship == 'pred_var':
        return pred_var_val, ml_dollars_test
    elif relationship == 'rmse':
        return rmse_val, ml_dollars_test
    elif relationship == 'ml_winrate':
        return ml_win_pct_val, ml_dollars_test
    elif relationship == 'ml_dollars':
        return ml_dollars_val, ml_dollars_test
    elif relationship == 'spread_winrate':
        return spread_win_pct_val, ml_dollars_test
    elif relationship == 'spread_dollars':
        return spread_dollars_val, ml_dollars_test
    else:
        print("Relationship ERROR")
        exit()


if __name__ == '__main__':
    # Path(__file__) returns absolute file path. Taking the .parent returns the folder this script is in. The .parent of
    # that is the main folder for this project which we wish to use as the main working directory.
    os.chdir(Path(__file__).parent.parent)

    # Print all rows and cols of dataframes
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    feat_file_name = [r"data\features\feat_simple_", r"data\features\feat_advanced_"]
    data_name = [r"data\processed\clean_data_simple_", r"data\processed\clean_data_advanced_"]
    model_type = ["simple", "advanced"]

    # for i in tqdm(range(len(feat_file_name))):
    i = 1
    prev_count = 80
    # for prev_count in tqdm(di.prev_count, colour='#ff00ff'):
    # Grabbing correct features/target file and cleaned data df for bet info
    feat_df = pd.read_csv(feat_file_name[i] + str(prev_count) + '.csv')
    data_df = pd.read_csv(data_name[i] + str(prev_count) + '.csv')

    for relation in tqdm(['mae', 'med', 'ex_var', 'pred_var', 'rmse',
                          'ml_winrate', 'ml_dollars', 'spread_winrate', 'spread_dollars'], colour='#8000ff'):
        ml_ud = pd.DataFrame()
        studies = pd.DataFrame()

        # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
        for yr in tqdm(di.test_years, colour='#00ff00'):
            # The season we will be testing and betting on. One year prior to this is the season we train until.
            end_year = yr
            # Initially use 2012 but can test a rolling 2-4 yrs to combat concept drift.
            # Season we begin training our data on.
            start_year = end_year - 3

            # Splitting features and targets into training and testing based on year
            X_test = feat_df.loc[feat_df.season == end_year].copy()
            y_test = X_test.pop('pts_home')
            # Betting line for a single team's points as a baseline model
            vegas_y_test = X_test.pop('ou_home')

            X = feat_df.loc[((feat_df.season < end_year) & (feat_df.season >= start_year))].copy()
            y = X.pop('pts_home')
            vegas_y = X.pop('ou_home')

            # All betting lines for the training years
            bet_df = bet.bet_info(data_df.loc[((data_df.season < end_year) & (data_df.season >= start_year))]).copy()
            bet_df.reset_index(inplace=True, drop=True)
            # All betting lines for the test year
            bet_df_test = bet.bet_info(data_df.loc[data_df.season == end_year])
            bet_df_test.reset_index(inplace=True, drop=True)

            for split_df in [X, y]:
                split_df.reset_index(inplace=True, drop=True)

            # Set index of bet_df to match the alternating one of X and y so that split will split the same rows.
            # Split doesnt split same rows if row indexes are different
            bet_df.index = X[::2].index
            X_train, X_val, y_train, y_val, bet_df_train, bet_df_val = \
                model_selection.train_test_split(X[::2], y[::2], bet_df, test_size=0.2, random_state=12)

            # Get the training rows and their pair store and sort them in an np.array
            temp = np.concatenate((X_train.index.values, X_train.index.values + 1))
            temp = np.sort(temp)
            # Then select the actual training set with paired rows
            X_train = X.iloc[temp]
            y_train = y.iloc[temp]

            # Do the same for validation set
            temp = np.concatenate((X_val.index.values, X_val.index.values + 1))
            temp = np.sort(temp)
            # Then select the actual training set with paired rows
            X_val = X.iloc[temp]
            y_val = y.iloc[temp]

            # Sort bet_dfs
            bet_df_train = bet_df_train.sort_index(inplace=False).copy()
            bet_df_val = bet_df_val.sort_index(inplace=False).copy()

            # Reset all row indices of split dfs
            for split_df in [X_train, X_val, y_train, y_val, bet_df_train, bet_df_val, X_test, y_test]:
                split_df.reset_index(inplace=True, drop=True)

            var_rem = feature_selection.VarianceThreshold(0.02).fit(X_train)
            X_train = var_rem.transform(X_train)
            X_val = var_rem.transform(X_val)
            X_test = var_rem.transform(X_test)

            # Scale features. Use MaxAbs since we do not need features to have mean 0. Also makes more intuitive sense
            # as none of the stats can be negative.
            # Loop here to test different preprocessing.
            # pre = [preprocessing.MaxAbsScaler(), preprocessing.MinMaxScaler(), preprocessing.MaxAbsScaler()]
            # for p in pre:
            scaler = preprocessing.PowerTransformer().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Set up a partial function with our own passed arguments into the optimization objective func
            objective_partial = partial(objective,
                                        train_x=X_train, train_y=y_train, train_bet=bet_df_train,
                                        val_x=X_val, val_y=y_val, val_bet=bet_df_val,
                                        test_x=X_test, test_y=y_test, test_bet=bet_df_test, relationship=relation
                                        )
            # Create study and optimize

            search_space = {"reg_lambda": [0.01, 0.3],
                            # L1 regularization weight.
                            "reg_alpha": [0.01, 0.3],
                            # sampling ratio for training data.
                            "subsample": [0.6, 0.9],
                            # sampling according to each tree.
                            "colsample_bytree": [0.2, 0.45, .7],
                            "max_depth": [2, 5, 8],
                            }

            if relation == 'mae' or relation == 'med' or relation == 'pred_var' or relation == 'rmse':
                study = optuna.create_study(directions=['minimize', 'maximize'],
                                            sampler=optuna.samplers.GridSampler(search_space))
            else:
                study = optuna.create_study(directions=['maximize', 'maximize'],
                                            sampler=optuna.samplers.GridSampler(search_space))
            study.optimize(objective_partial, n_trials=80, timeout=2000, n_jobs=1, gc_after_trial=True)

            # Append result values to studies to aggregate them all across years
            for trial_num in range(len(study.trials)):
                studies = studies.append(pd.Series(study.trials[trial_num].values).transpose(), ignore_index=True)

        if relation == 'mae':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['MAE_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='MAE_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Mean Absolute Error of Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\mae_dollars.png'))
            print('SAVED:' + str('models\\graphs\\' + model_type[i] + r'\optuna_obj\mae_dollars.png'))
            fig.clf()
        elif relation == 'med':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['MED_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='MED_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Median Absolute Error of Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\med_dollars.png'))
            print('SAVED:' + str('models\\graphs\\' + model_type[i] + r'\optuna_obj\med_dollars.png'))
            fig.clf()
        elif relation == 'ex_var':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['Ex_Var_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='Ex_Var_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Explained Variance of Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\exvar_dollars.png'))
            fig.clf()
        elif relation == 'pred_var':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['Pred_Var_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='Pred_Var_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Prediction Variance of Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\predvar_dollars.png'))
            fig.clf()
        elif relation == 'rmse':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['RMSE_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='RMSE_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Root Mean Squared Error of Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\rmse_dollars.png'))
            fig.clf()
        elif relation == 'ml_winrate':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['ml_Winrate_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='ml_Winrate_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Moneyline Win Rate of Validation Set  vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\ml_winrate_dollars.png'))
            fig.clf()
        elif relation == 'ml_dollars':
            studies.columns = ['ml_Dollars_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='ml_Dollars_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            fig.suptitle(r"Moneyline Dollars Earned on Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\ml_dollars_dollars.png'))
            fig.clf()
        elif relation == 'spread_winrate':
            # Create plot of performance objective on validation set and money made from betting on test set.
            studies.columns = ['spread_Winrate_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='spread_Winrate_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            # need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
            fig.suptitle(r"Spread Win Rate of Validation Set  vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\spread_winrate_dollars.png'))
            fig.clf()
        elif relation == 'spread_dollars':
            studies.columns = ['spread_Dollars_Validation', 'Net_Dollars_Test']
            fig, ax = plt.subplots(figsize=(12, 10))
            ax = sns.regplot(data=studies, x='spread_Dollars_Validation', y='Net_Dollars_Test', ci=80, truncate=False)
            fig.suptitle(r"Spread Dollars Earned on Validation Set vs. Dollars Earned from Betting on Test Set")
            fig.savefig(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\spread_dollars_dollars.png'))
            fig.clf()
        else:
            print("Graph Error")
            exit()
