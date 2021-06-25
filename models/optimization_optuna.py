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


def objective(trial, train_x, train_y, train_bet, val_x, val_y, val_bet, test_x, test_y, test_bet, rand_state):
    # parameter space to test
    # use pseudohuber since it can punish closer error values harder that further errors.
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
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0),
        "n_jobs": 6,
        "n_estimators": 50,
        "random_state": rand_state,
        "verbosity": 0,
        "base_score": 105
    }

    # Since XGBRegressor doesnt take a param argument rather just all the params as individual named arguments
    # use **kargs
    model_xgb = xgb.XGBRegressor(**param).fit(train_x, train_y)
    pred_val = model_xgb.predict(val_x)

    mae_val = metrics.mean_absolute_error(val_y, pred_val)

    rmse_val = metrics.mean_squared_error(val_y, pred_val, squared=False)

    predvar_val = pred_val.var()

    exvar_val = metrics.explained_variance_score(val_y, pred_val)

    spread_outcome_val, spread_win_pct_val, spread_dollars_val = \
        bet.spread_outcome(pred_val, val_bet.spread, val_bet.home_margin,
                           margin=1)

    ml_outcome_val, ml_win_pct_val, ml_dollars_val = \
        bet.ml_outcome(pred_val, val_bet.loc[:, ['ml_home', 'ml_away']],
                       val_bet.home_win, margin=0, lower_bound_odds=0)

    return mae_val, exvar_val


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

    for yr_type in ['all_yrs']:
        mae_df = pd.DataFrame()
        exvar_df = pd.DataFrame()

        # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
        for yr in tqdm(di.test_years, colour='#00ff00'):
            best_trials_df = pd.DataFrame()
            mae_list = []
            exvar_list = []

            # The season we will be testing and betting on. One year prior to this is the season we train until.
            end_year = yr
            # Initially use 2012 but can test a rolling 2-4 yrs to combat concept drift.
            # Season we begin training our data on.
            if yr_type == 'all_yrs':
                start_year = 2012
            else:
                start_year = end_year-3

            # Splitting features and targets into training and testing based on year
            X_test = feat_df.loc[feat_df.season == end_year]
            y_test = X_test.pop('pts_home')
            # Betting line for a single team's points as a baseline model
            vegas_y_test = X_test.pop('ou_home')

            X = feat_df.loc[((feat_df.season < end_year) & (feat_df.season >= start_year))]
            y = X.pop('pts_home')
            vegas_y = X.pop('ou_home')

            # All betting lines for the training years
            bet_df = bet.bet_info(data_df.loc[((data_df.season < end_year) & (data_df.season >= start_year))])
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
                model_selection.train_test_split(X[::2], y[::2], bet_df, test_size=0.3, random_state=12)

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

            var_rem = feature_selection.VarianceThreshold(0).fit(X_train)
            X_train = var_rem.transform(X_train)
            X_val = var_rem.transform(X_val)
            X_test = var_rem.transform(X_test)

            # Scale features. Use MaxAbs since we do not need features to have mean 0. Also makes more intuitive sense
            # as none of the stats can be negative.
            # Loop here to test different preprocessing.
            # pre = [preprocessing.MaxAbsScaler(), preprocessing.MinMaxScaler(), preprocessing.MaxAbsScaler()]
            # for p in pre:
            scaler_qt = preprocessing.QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                                          random_state=12).fit(X_train)
            X_train = scaler_qt.transform(X_train)
            X_val = scaler_qt.transform(X_val)
            X_test = scaler_qt.transform(X_test)
            scaler_ss = preprocessing.StandardScaler(with_mean=True).fit(X_train)
            X_train = scaler_ss.transform(X_train)
            X_val = scaler_ss.transform(X_val)
            X_test = scaler_ss.transform(X_test)

            for rand_int in tqdm([12, 165903, 438567, 38376, 87756]):
                # Set up a partial function with our own passed arguments into the optimization objective func
                objective_partial = partial(objective,
                                            train_x=X_train, train_y=y_train, train_bet=bet_df_train,
                                            val_x=X_val, val_y=y_val, val_bet=bet_df_val,
                                            test_x=X_test, test_y=y_test, test_bet=bet_df_test, rand_state=rand_int
                                            )
                # Create study and optimize
                study = optuna.create_study(directions=['minimize', 'maximize'],
                                            sampler=optuna.samplers.NSGAIISampler(seed=12))
                study.optimize(objective_partial, n_trials=100, timeout=2000, n_jobs=1, gc_after_trial=True)

                # Store the best trial(s) in a df
                trials = study.best_trials
                for trial_num in range(len(trials)):
                    print("Value: {}".format(trials[trial_num].values))
                    mae_list.append(trials[trial_num].values[0])
                    exvar_list.append(trials[trial_num].values[1])
                    best_trials_df = best_trials_df.append(pd.DataFrame(list(trials[trial_num].params.items())),
                                                           ignore_index=True)

            mae_df.loc['MAE', str(end_year)] = np.mean(mae_list)
            exvar_df.loc['ExVar', str(end_year)] = np.mean(exvar_list) * 100

            best_trials_df.columns = ['param', 'value']
            # Will turn params into column names and average their values
            best_trials_df = best_trials_df.pivot_table(index=best_trials_df.index // len(best_trials_df),
                                              columns='param',
                                              values='value')
            best_trials_df.to_csv(r'models/optuna/advanced/' + str(prev_count) + r'/' + yr_type + r'/' + str(yr) + '.csv', index=False)

            # Append result values to studies to aggregate them all across years
            #for trial_num in range(len(study.trials)):
                #studies = studies.append(pd.Series(study.trials[trial_num].values).transpose(), ignore_index=True)

            '''
            print("Number of finished trials: ", len(study.trials))
            print("Best trial:")
            trials = study.best_trials
            # Change back to trial.value if only 1 objective
            for trial_num in len(trials):
                print("  Value: {}".format(trials[trial_num].values))
                print("  Params: ")
                for key, value in trials[trial_num].params.items():
                    print("    {}: {}".format(key, value))
    
            # Can also do studies.add_trials on previous study
            fig = optuna.visualization.plot_pareto_front(study, target_names=["mae_val", "ex_var_val", "dollars_test"])
            fig.write_html(str('models\\graphs\\' + model_type[i] + r'\optuna_obj\multiopt' + str(yr) + '.html'))
            '''
        mae_df['all_years'] = mae_df.mean(axis=1)
        mae_df.to_csv(r'models/optuna/advanced/' + str(prev_count) + r'/' + yr_type + r'/mae.csv')
        exvar_df['all_years'] = exvar_df.mean(axis=1)
        exvar_df.to_csv(r'models/optuna/advanced/' + str(prev_count) + r'/' + yr_type + r'/exvar.csv')

