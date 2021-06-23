import numpy as np
import pandas as pd

from util import train as hp
import util.dictionaries as di

import os
from pathlib import Path

from multiprocessing import Pool
from tqdm import tqdm
from joblib import dump
import copy
from itertools import compress

from sklearn import preprocessing, model_selection, metrics

if __name__ == '__main__':
    # Path(__file__) returns absolute file path. Taking the .parent returns the folder this script is in. The .parent of
    # that is the main folder for this project which we wish to use as the main working directory.
    os.chdir(Path(__file__).parent.parent)

    # Print all rows and cols of dataframes
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    feat_file_name = [r"data\features\feat_simple_", r"data\features\feat_advanced_"]
    model_type = ["simple", "advanced"]

    for i in tqdm(range(len(feat_file_name))):
        prev_count = 80
        # for prev_count in tqdm(di.prev_count, colour='#ff00ff'):
        # Grabbing correct features/target file
        feat_df = pd.read_csv(feat_file_name[i] + str(prev_count) + '.csv')

        # DataFrame to store error
        error = pd.DataFrame()

        # Initialize the algorithms which will be used to train models
        algos_raw = copy.deepcopy(di.algos_raw)

        # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
        for yr in tqdm(di.test_years, colour='#00ff00'):
            # The season we will be testing and betting on. One year prior to this is the season we train until.
            end_year = yr
            # Initially use 2012 but can test a rolling 2-4 yrs to combat concept drift.
            # Season we begin training our data on.
            start_year = 2012

            # Create file names to store the trained models. Will just be model_name and test year.
            # Create a new list of names as to not modify our import di.algos_name
            algos_name = []
            for name in range(len(di.algos_name)):
                algos_name.append(di.algos_name[name] + '_' + str(end_year))

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

            # Set index of bet_df to match the alternating one of X so that split will split the same rows.
            # Split doesnt split same rows if row indexes are different
            X_train, X_val = \
                model_selection.train_test_split(X[::2], test_size=0.3, random_state=12)

            # Get the training rows and their pair store and sort them in an np.array
            temp = np.concatenate((X_train.index.values, X_train.index.values + 1))
            temp = np.sort(temp)
            # Then select the actual training set of X, y, and vegas_y with paired rows
            X_train = X.iloc[temp]
            y_train = y.iloc[temp]
            vegas_y_train = vegas_y[temp]

            # Do the same for validation set
            temp = np.concatenate((X_val.index.values, X_val.index.values + 1))
            temp = np.sort(temp)
            X_val = X.iloc[temp]
            y_val = y.iloc[temp]
            vegas_y_val = vegas_y[temp]

            # Reset all row indices of split dfs
            for split_df in [X_train, X_val, y_train, y_val, vegas_y_train, vegas_y_val]:
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

            algo_split = np.array_split(algos_raw, 3)
            pool = Pool(3)
            models_fitted = pool.starmap(hp.fit_models, [
                (algo_split[0], X_train, y_train),
                (algo_split[1], X_train, y_train),
                (algo_split[2], X_train, y_train)
            ])
            # Store all trained models in a list, some are nested lists due to the way our splits and starmap works.
            # Hence the double loops through list and sublist.
            models = [obj for sublist in models_fitted for obj in sublist]
            # Use starmap as it takes a list of iterable tuples that are unpacked as arguments for the function.
            pool.close()
            pool.join()

            # Pickling and saving trained model objects
            for mod_num in range(len(models)):
                dump(models[mod_num], r'models/saved/' + model_type[i] + r'/' + str(prev_count) + r'/' +
                     algos_name[mod_num] + r'.pkl')

        error['total'] = error.mean(axis=1)
        print(error)
