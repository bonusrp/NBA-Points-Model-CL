import numpy as np
import pandas as pd

from util import bet
import util.dictionaries as di

import os
from pathlib import Path

from tqdm import tqdm
from joblib import load

from sklearn import preprocessing, model_selection, metrics, feature_selection
import xgboost as xgb


if __name__ == '__main__':
    # Path(__file__) returns absolute file path. Taking the .parent returns the folder this script is in. The .parent of
    # that is the main folder for this project which we wish to use as the main working directory.
    os.chdir(Path(__file__).parent.parent)

    # Print all rows and cols of dataframes
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.options.mode.chained_assignment = None

    feat_file_name = [r"data/features/feat_advanced_"]
    data_name = [r"data/processed/clean_data_advanced_"]
    model_type = ["optimal"]
    param_name = [r"models/optuna/advanced/80/all_yrs/"]

    i = 0
    prev_count = 80
    # for prev_count in tqdm(di.prev_count, colour='#ff00ff'):
    # Grabbing correct features/target file
    feat_df = pd.read_csv(feat_file_name[i] + str(prev_count) + '.csv')
    # Cleaned data which will be used to get vegas's betting lines to see if our model can make profitable bets
    data_df = pd.read_csv(data_name[i] + str(prev_count) + '.csv')
    ml_ud_all_master = pd.DataFrame()

    for low in tqdm(di.lower_bounds_opt, colour='#339CFF'):
        # List to store dataframes of money made on each type of bet
        ml_ud_master = []

        # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
        for yr in tqdm(di.test_years, colour='#00ff00'):
            ml_ud = pd.DataFrame()
            ml_ud_folds = pd.DataFrame()

            # The season we will be testing and betting on. One year prior to this is the season we train until.
            end_year = yr
            # Initially use 2012 but can test a rolling 2-4 yrs to combat concept drift.
            # Season we begin training our data on.
            start_year = 2012

            # Create file names to store the trained models into. Will just be model_name and test year.
            algos_name = di.algos_name_opt + '_' + str(end_year)

            # All betting lines for the correct year
            bet_df = bet.bet_info(data_df.loc[((data_df.season < end_year) & (data_df.season >= start_year))])
            bet_df.reset_index(inplace=True, drop=True)

            # Splitting features and targets into training and testing based on year
            X_test = feat_df.loc[feat_df.season == end_year]
            y_test = X_test.pop('pts_home')
            # Betting line for a single team's points as a baseline model
            vegas_y_test = X_test.pop('ou_home')

            X = feat_df.loc[((feat_df.season < end_year) & (feat_df.season >= start_year))]
            y = X.pop('pts_home')
            vegas_y = X.pop('ou_home')

            # Reset all row indices of split dfs
            for split_df in [X, y, vegas_y_test, X_test, y_test, vegas_y]:
                split_df.reset_index(inplace=True, drop=True)

            # Lists to store objectives through kfolds
            mlud_list_cv = []

            # Initialize folds
            num_folds = di.num_folds_opt
            kf = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=12)
            # Dict and lists datasets through kfolds
            X_train_l = []
            X_val_l = []
            y_train_l = []
            y_val_l = []
            bet_df_val_l = []

            for train_index, test_index in kf.split(X[::2]):
                # Get the training rows and their pair store and sort them in an np.array
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
                for split_df in [X_train, X_val, y_train, y_val, X_test, y_test, bet_df_train, bet_df_val]:
                    split_df.reset_index(inplace=True, drop=True)
                # Append to the lists storing folds
                X_train_l.append(X_train)
                X_val_l.append(X_val)
                y_train_l.append(y_train)
                y_val_l.append(y_val)
                bet_df_val_l.append(bet_df_val)

            # Initialize dict after all datasets are stored into their respective lists
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

                scaler_qt = preprocessing.QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                                              random_state=12).fit(X_train)
                X_train = scaler_qt.transform(X_train)
                X_val = scaler_qt.transform(X_val)
                scaler_ss = preprocessing.StandardScaler(with_mean=True).fit(X_train)
                X_train = scaler_ss.transform(X_train)
                X_val = scaler_ss.transform(X_val)

                # Parameters file
                param_df = pd.read_csv(param_name[i] + str(end_year) + '.csv')
                pred_df = pd.DataFrame()
                for rand_int in [12, 165903, 438567, 38376, 87756]:
                    model_xgb = xgb.XGBRegressor(objective='reg:pseudohubererror',
                                                 colsample_bytree=np.round(param_df.colsample_bytree[0], 2),
                                                 learning_rate=np.round(param_df.learning_rate[0], 3),
                                                 max_depth=int(np.round(param_df.max_depth[0], 0)),
                                                 reg_alpha=np.round(param_df.reg_alpha[0], 5),
                                                 reg_lambda=np.round(param_df.reg_lambda[0], 5),
                                                 subsample=np.round(param_df.subsample[0], 2),
                                                 n_jobs=6, n_estimators=50,
                                                 eval_metric='mae', tree_method='hist', booster='gbtree',
                                                 base_score=105,
                                                 verbosity=0, random_state=rand_int).fit(X_train, y_train)
                    # print(model_xgb.get_booster().get_score(importance_type="gain"))
                    # Get predictions of testing set's target value using testing set's features
                    pred_df.loc[:, str(rand_int)] = pd.Series(model_xgb.predict(X_val))
                # Get median of all predictions and evaluate
                prediction = pred_df.mean(axis=1)

                # Calculate money made on three types of bets for each model across all margin thresholds
                for threshold in di.margins_opt:
                    # For ml_bets on underdogs only, so only positive odds
                    ml_outcome, ml_win_pct, ml_dollars = \
                        bet.ml_outcome(prediction, bet_df_val.loc[:, ['ml_home', 'ml_away']],
                                       bet_df_val.home_win, margin=threshold, lower_bound_odds=low)
                    ml_ud.loc[fold_id, str(threshold)] = ml_dollars

            # Append a df of dollars earned for each type of bet for each model and each threshold for this specific yr.
            ml_ud_folds = ml_ud.mean(axis=0)
            ml_ud_master.append(pd.DataFrame(ml_ud_folds).T)

        # Saving year-by-year and aggregate money made for SPREAD bets to csv
        # Creating a df to store the total money earned across all years.
        totals_ml_ud = pd.DataFrame(index=ml_ud_master[0].index.values, columns=ml_ud_master[0].columns.values)
        totals_ml_ud.fillna(0, inplace=True)

        # Loop though each year's df and results to a csv while also calculating the total
        for yr_num in range(len(ml_ud_master)):
            totals_ml_ud = ml_ud_master[yr_num] + totals_ml_ud
            ml_ud_master[yr_num].to_csv(r'models/results/' + model_type[i] + r'/' + str(prev_count) + r'/' +
                                        'ml_money_ud_' + str(yr_num) + r'.csv')

        # Unstack the thresholds which were columns names
        totals_ml_ud = totals_ml_ud.stack().reset_index(drop=False)
        totals_ml_ud.columns = ['model', 'threshold', 'dollars']
        totals_ml_ud['lower_odds'] = low
        # Append to the df which stores all runs through lower and thresholds
        ml_ud_all_master = ml_ud_all_master.append(totals_ml_ud)

    ml_ud_all_master.to_csv(r'models/optuna/advanced/80/opt_mlud.csv', index=False)
