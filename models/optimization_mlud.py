import numpy as np
import pandas as pd

from util import bet
import util.dictionaries as di

import os
from pathlib import Path

from tqdm import tqdm
from joblib import load

from sklearn import preprocessing, model_selection, metrics

if __name__ == '__main__':
    # Path(__file__) returns absolute file path. Taking the .parent returns the folder this script is in. The .parent of
    # that is the main folder for this project which we wish to use as the main working directory.
    os.chdir(Path(__file__).parent.parent)

    # Print all rows and cols of dataframes
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.options.mode.chained_assignment = None

    feat_file_name = [r"data\features\feat_simple_", r"data\features\feat_advanced_"]
    data_name = [r"data\processed\clean_data_simple_", r"data\processed\clean_data_advanced_"]
    model_type = ["simple", "advanced"]

    i = 1
    prev_count = 80
    # for prev_count in tqdm(di.prev_count, colour='#ff00ff'):
    # Grabbing correct features/target file
    feat_df = pd.read_csv(feat_file_name[i] + str(prev_count) + '.csv')
    # Cleaned data which will be used to get vegas's betting lines to see if our model can make profitable bets
    data_df = pd.read_csv(data_name[i] + str(prev_count) + '.csv')
    ml_ud_all_master = pd.DataFrame()

    for low in di.lower_bounds_opt:
        # List to store dataframes of money made on each type of bet
        ml_ud_master = []

        # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
        for yr in tqdm(di.test_years, colour='#00ff00'):
            ml_ud = pd.DataFrame()

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

            bet_df.index = X[::2].index
            X_train, X_val, bet_df_train, bet_df_val = \
                model_selection.train_test_split(X[::2], bet_df, test_size=0.3, random_state=12)

            # Get the training rows and their pair store and sort them in an np.array
            temp = np.concatenate((X_train.index.values, X_train.index.values + 1))
            temp = np.sort(temp)
            X_train = X.iloc[temp]
            y_train = y.iloc[temp]
            vegas_y_train = vegas_y[temp]

            # Do the same for validation set
            temp = np.concatenate((X_val.index.values, X_val.index.values + 1))
            temp = np.sort(temp)
            X_val = X.iloc[temp]
            y_val = y.iloc[temp]
            vegas_y_val = vegas_y[temp]

            # Sort bet_dfs
            bet_df_train = bet_df_train.sort_index(inplace=False).copy()
            bet_df_val = bet_df_val.sort_index(inplace=False).copy()

            for split_df in [X_train, X_val, y_train, y_val, vegas_y_train, vegas_y_val, bet_df_train, bet_df_val]:
                split_df.reset_index(inplace=True, drop=True)

            # Scale features.
            # Loop here to test different preprocessing.
            # pre = [preprocessing.MaxAbsScaler(), preprocessing.MinMaxScaler(), preprocessing.MaxAbsScaler()]
            # for p in pre:
            scaler_qt = preprocessing.QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                                          random_state=12).fit(X_train)
            X_train = scaler_qt.transform(X_train)
            X_val = scaler_qt.transform(X_val)
            scaler_ss = preprocessing.StandardScaler(with_mean=True).fit(X_train)
            X_train = scaler_ss.transform(X_train)
            X_val = scaler_ss.transform(X_val)

            # Unpickling all the trained models in read only mode and keeping them all in a list
            models = load(r'models/saved/' + model_type[i] + r'/' + str(prev_count) + r'/' +
                          algos_name + r'.pkl')

            # Get predictions of testing set's target value using testing set's features
            prediction = models.predict(X_val)

            # Calculate money made on three types of bets for each model across all margin thresholds
            for threshold in di.margins_opt:
                # For ml_bets on underdogs only, so only positive odds
                ml_outcome, ml_win_pct, ml_dollars = \
                    bet.ml_outcome(prediction, bet_df_val.loc[:, ['ml_home', 'ml_away']],
                                   bet_df_val.home_win, margin=threshold, lower_bound_odds=low)
                ml_ud.loc[di.algos_name_opt, str(threshold)] = ml_dollars

            # Append a df of dollars earned for each type of bet for each model and each threshold for this specific yr.
            ml_ud_master.append(ml_ud)

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
