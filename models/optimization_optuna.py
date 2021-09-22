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


def objective(trial, X, y, X_test, y_test, bet_df, bet_df_test):
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
        "random_state": 12,
        "verbosity": 0,
        "base_score": 105
    }

    # Since XGBRegressor doesnt take a param argument rather just all the params as individual named arguments
    # use **kargs

    # Lists to store objectives through kfolds
    mae_list_cv = []
    exvar_list_cv = []
    mlud_list_cv = []

    # Initialize folds
    num_folds = di.num_folds_opt
    # Can use shuffle which is still unrealistic but more realistic than training on 2014 and testing on 2012
    kf = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=12)
    # Dict and lists datasets through kfolds
    X_train_l = []
    X_val_l = []
    y_train_l = []
    y_val_l = []
    bet_df_val_l = []

    for train_index, test_index in kf.split(X[::2]):
        # Get the training rows and their pair store and sort them in an np.array
        # Need to multiply by 2 because kf.split renumbers X[::2] and returns new index numbers while
        # train_test_split returned the actual rows with the true row indexes intact so need to extract in a diff manner
        temp = np.concatenate((train_index * 2, train_index * 2 + 1))
        temp = np.sort(temp)
        # Then select the actual training set with paired rows
        X_train = X.iloc[temp, :]
        y_train = y.iloc[temp]

        # Do the same for validation set
        temp = np.concatenate((test_index * 2, test_index * 2 + 1))
        temp = np.sort(temp)
        # Then select the actual training set with paired rows
        X_val = X.iloc[temp, :]
        y_val = y.iloc[temp]

        # Sort bet_dfs
        bet_df_train = bet_df.iloc[train_index, :].copy()
        bet_df_val = bet_df.iloc[test_index, :].copy()

        # Reset all row indices of split dfs
        for split_df in [X_train, X_val, y_train, y_val, X_test, y_test, bet_df_train, bet_df_val, bet_df_test]:
            split_df.reset_index(inplace=True, drop=True)
        # Append to the lists storing folds
        X_train_l.append(X_train)
        X_val_l.append(X_val)
        y_train_l.append(y_train)
        y_val_l.append(y_val)
        bet_df_val_l.append(bet_df_val)

    # Initialize dict after all datasets are stored into their respective lists. Do this here because dicts initialize
    # with lists as is and doesnt update if lists change unless dict is reconstructed.
    kfold_splits = {'X_train': X_train_l, 'X_val': X_val_l,
                    'y_train': y_train_l, 'y_val': y_val_l,
                    'bet_df_val': bet_df_val_l}
    for fold_id in range(num_folds):
        # Grabbing the correct fold datasets
        X_train = kfold_splits['X_train'][fold_id]
        X_val = kfold_splits['X_val'][fold_id]
        y_train = kfold_splits['y_train'][fold_id]
        y_val = kfold_splits['y_val'][fold_id]
        bet_df_val = kfold_splits['bet_df_val'][fold_id]

        # Remove no variance features
        var_rem = feature_selection.VarianceThreshold(0).fit(X_train)
        X_train = var_rem.transform(X_train)
        X_val = var_rem.transform(X_val)

        # Scale features
        scaler_qt = preprocessing.QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                                      random_state=12).fit(X_train)
        X_train = scaler_qt.transform(X_train)
        X_val = scaler_qt.transform(X_val)
        scaler_ss = preprocessing.StandardScaler(with_mean=True).fit(X_train)
        X_train = scaler_ss.transform(X_train)
        X_val = scaler_ss.transform(X_val)

        model_xgb = xgb.XGBRegressor(**param).fit(X_train, y_train)
        pred_val = model_xgb.predict(X_val)

        mae_val = metrics.mean_absolute_error(y_val, pred_val)
        mae_list_cv.append(mae_val)

        exvar_val = metrics.explained_variance_score(y_val, pred_val)
        exvar_list_cv.append(exvar_val)


    return np.mean(mae_list_cv), np.mean(exvar_list_cv)


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

    for yr_type in ['all_yrs']:  # , 'three_yrs']:
        mae_df = pd.DataFrame()
        exvar_df = pd.DataFrame()

        # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
        for yr in tqdm(di.test_years, colour='#00ff00'):
            best_trials_df = pd.DataFrame()
            mae_list_yr = []
            exvar_list_yr = []

            # The season we will be testing and betting on. One year prior to this is the season we train until.
            end_year = yr
            # Initially use 2012 but can test a rolling 2-4 yrs to combat concept drift.
            # Season we begin training our data on.
            if yr_type == 'all_yrs':
                start_year = 2012
            else:
                start_year = end_year - 3

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

            objective_partial = partial(objective,
                                        X=X, y=y,
                                        X_test=X_test, y_test=y_test,
                                        bet_df=bet_df, bet_df_test=bet_df_test
                                        )

            # Create study and optimize
            study = optuna.create_study(directions=['minimize', 'maximize'],
                                        sampler=optuna.samplers.NSGAIISampler(seed=12))
            study.optimize(objective_partial, n_trials=90, timeout=2000, n_jobs=1, gc_after_trial=True)

            # Store the best trial(s) in a df
            trials = study.best_trials
            for trial_num in range(len(trials)):
                print("Value: {}".format(trials[trial_num].values))
                mae_list_yr.append(trials[trial_num].values[0])
                exvar_list_yr.append(trials[trial_num].values[1])
                best_trials_df = best_trials_df.append(pd.DataFrame(list(trials[trial_num].params.items())),
                                                       ignore_index=True)

            mae_df.loc['MAE', str(end_year)] = np.mean(mae_list_yr)
            exvar_df.loc['ExVar', str(end_year)] = np.mean(exvar_list_yr) * 100

            best_trials_df.columns = ['param', 'value']
            # Will turn params into column names and average their values
            best_trials_df = best_trials_df.pivot_table(index=best_trials_df.index // len(best_trials_df),
                                                        columns='param',
                                                        values='value')
            best_trials_df.to_csv(
                r'models/optuna/advanced/' + str(prev_count) + r'/' + yr_type + r'/' + str(yr) + '.csv', index=False)

            # Append result values to studies to aggregate them all across years
            # for trial_num in range(len(study.trials)):
            # studies = studies.append(pd.Series(study.trials[trial_num].values).transpose(), ignore_index=True)

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

