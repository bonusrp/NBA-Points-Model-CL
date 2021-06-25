import pandas as pd
import numpy as np

from util import bet
import util.dictionaries as di

import os
from pathlib import Path

from tqdm import tqdm
from joblib import dump

from sklearn import preprocessing, metrics
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

    # DataFrame to store various error and metrics
    mae_error = pd.DataFrame()
    mse_error = pd.DataFrame()
    med_error = pd.DataFrame()
    explained_var = pd.DataFrame()
    pred_var = pd.DataFrame()
    # List to store dataframes of money made on each type of bet
    spread_master = []
    ou_master = []
    ml_all_master = []
    ml_ud_dol_master = []
    ml_ud_wr_master = []

    # Loop for rolling year train, test, bet -> retrain, test, bet -> repeat
    for yr in tqdm(di.test_years, colour='#00ff00'):
        spread = pd.DataFrame()
        ou = pd.DataFrame()
        ml_all = pd.DataFrame()
        ml_ud_dol = pd.DataFrame()
        ml_ud_wr = pd.DataFrame()

        # The season we will be testing and betting on. One year prior to this is the season we train until.
        end_year = yr
        # Initially use 2012 but can test a rolling 2-4 yrs to combat concept drift.
        # Season we begin training our data on.
        start_year = 2012

        # All betting lines for the correct year
        bet_df = bet.bet_info(data_df.loc[data_df.season == end_year])
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
        for split_df in [X, y, X_test, y_test, vegas_y, vegas_y_test]:
            split_df.reset_index(inplace=True, drop=True)

        # Scale features.
        scaler_qt = preprocessing.QuantileTransformer(n_quantiles=1000, output_distribution='normal',
                                                      random_state=12).fit(X)
        X = scaler_qt.transform(X)
        X_test = scaler_qt.transform(X_test)
        scaler_ss = preprocessing.StandardScaler(with_mean=True).fit(X)
        X = scaler_ss.transform(X)
        X_test = scaler_ss.transform(X_test)

        # Parameters file
        param_df = pd.read_csv(param_name[i] + str(end_year) + '.csv')
        # Train multiple models on different seeds and average the predictions
        pred_df = pd.DataFrame()
        for rand_int in tqdm([12, 165903, 438567, 38376, 87756]):
            model_xgb = xgb.XGBRegressor(objective='reg:pseudohubererror',
                                         colsample_bytree=np.round(param_df.colsample_bytree[0], 2),
                                         learning_rate=np.round(param_df.learning_rate[0], 3),
                                         max_depth=int(np.round(param_df.max_depth[0], 0)),
                                         reg_alpha=np.round(param_df.reg_alpha[0], 5),
                                         reg_lambda=np.round(param_df.reg_lambda[0], 5),
                                         subsample=np.round(param_df.subsample[0], 2),
                                         n_jobs=6, n_estimators=50,
                                         eval_metric='mae', tree_method='hist', booster='gbtree', base_score=105,
                                         verbosity=0, random_state=rand_int).fit(X, y)
            # print(model_xgb.get_booster().get_score(importance_type="gain"))
            # Get predictions of testing set's target value using testing set's features
            pred_df.loc[:, str(rand_int)] = pd.Series(model_xgb.predict(X_test))

        # Get median of all predictions and evaluate
        prediction = pred_df.mean(axis=1)

        # Calculating prediction variance
        pred_var.loc['Vegas', str(end_year)] = vegas_y_test.var()
        pred_var.loc['XGB', str(end_year)] = prediction.var()

        # Get the performance metrics for vegas's models
        mae_error.loc['Vegas', str(end_year)] = metrics.mean_absolute_error(y_test, vegas_y_test)
        mse_error.loc['Vegas', str(end_year)] = metrics.mean_squared_error(y_test, vegas_y_test, squared=False)
        med_error.loc['Vegas', str(end_year)] = metrics.median_absolute_error(y_test, vegas_y_test)
        explained_var.loc['Vegas', str(end_year)] = metrics.explained_variance_score(y_test, vegas_y_test)
        pred_var.loc['Vegas', str(end_year)] = vegas_y_test.var()
        # Loop through my models and get performance metrics of each model's predictions vs. ground truth.
        mae_error.loc['XGB', str(end_year)] = \
            metrics.mean_absolute_error(y_test, prediction)
        mse_error.loc['XGB', str(end_year)] = \
            metrics.mean_squared_error(y_test, prediction, squared=False)
        med_error.loc['XGB', str(end_year)] = \
            metrics.median_absolute_error(y_test, prediction)
        explained_var.loc['XGB', str(end_year)] = \
            metrics.explained_variance_score(y_test, prediction)

        # Calculate money made on three types of bets for each model across all margin thresholds
        for threshold in di.margins_test:
            # For spread_bets
            spread_outcome, spread_win_pct, spread_dollars = \
                bet.spread_outcome(prediction, bet_df.spread, bet_df.home_margin,
                                   margin=threshold)
            # Saves a row of dollars earned for each threshold
            spread.loc['XGB', str(threshold)] = spread_dollars

            # For ou_bets
            ou_outcome, ou_win_pct, ou_dollars = \
                bet.ou_outcome(prediction, bet_df.ou_total, bet_df.points_total,
                               margin=threshold)
            ou.loc['XGB', str(threshold)] = ou_dollars
            # For ml_bets on all
            ml_outcome, ml_win_pct, ml_dollars = \
                bet.ml_outcome(prediction, bet_df.loc[:, ['ml_home', 'ml_away']],
                               bet_df.home_win, margin=threshold, lower_bound_odds=float('-inf'))
            ml_all.loc['XGB', str(threshold)] = ml_dollars

            # For ml_bets on underdogs only, so only positive odds
            ml_outcome, ml_win_pct, ml_dollars = \
                bet.ml_outcome(prediction, bet_df.loc[:, ['ml_home', 'ml_away']],
                               bet_df.home_win, margin=threshold, lower_bound_odds=di.lower_bounds_test)
            ml_ud_dol.loc['XGB', str(threshold)] = ml_dollars
            ml_ud_wr.loc['XGB', str(threshold)] = ml_win_pct

        # Append a df of dollars earned for each type of bet for each model and each threshold for this specific yr.
        spread_master.append(spread)
        ou_master.append(ou)
        ml_all_master.append(ml_all)
        ml_ud_dol_master.append(ml_ud_dol)
        ml_ud_wr_master.append(ml_ud_wr)

    # Saving year-by-year and mean aggregate prediction variance
    pred_var['total'] = pred_var.mean(axis=1)
    pred_var.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                    'pred_var' + r'.csv')

    # Saving year-by-year and mean aggregate error to csv
    mae_error['total'] = mae_error.mean(axis=1)
    mae_error.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                     'mae' + r'.csv')
    mse_error['total'] = mse_error.mean(axis=1)
    mse_error.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                     'rmse' + r'.csv')
    med_error['total'] = med_error.mean(axis=1)
    med_error.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                     'med' + r'.csv')
    explained_var['total'] = explained_var.median(axis=1)
    explained_var.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                         'ex_var' + r'.csv')

    # Saving year-by-year and aggregate money made for SPREAD bets to csv
    # Creating a df to store the total money earned across all years.
    totals_spread = pd.DataFrame(index=spread_master[0].index.values, columns=spread_master[0].columns.values)
    totals_spread.fillna(0, inplace=True)
    totals_ou = totals_spread.copy(deep=True)
    totals_ml_all = totals_spread.copy(deep=True)
    totals_ml_ud_dol = totals_spread.copy(deep=True)
    totals_ml_ud_wr = totals_spread.copy(deep=True)

    # Loop though each year's df and results to a csv while also calculating the total
    for yr_num in range(len(spread_master)):
        totals_spread = spread_master[yr_num] + totals_spread
        spread_master[yr_num].to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                                     'spread_money_' + str(yr_num) + r'.csv')
        totals_ou = ou_master[yr_num] + totals_ou
        ou_master[yr_num].to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                                 'ou_money_' + str(yr_num) + r'.csv')
        totals_ml_all = ml_all_master[yr_num] + totals_ml_all
        ml_all_master[yr_num].to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                                     'ml_money_all_' + str(yr_num) + r'.csv')
        totals_ml_ud_dol = ml_ud_dol_master[yr_num] + totals_ml_ud_dol
        ml_ud_dol_master[yr_num].to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                                    'ml_money_ud_' + str(yr_num) + r'.csv')
        totals_ml_ud_wr = ml_ud_wr_master[yr_num] + totals_ml_ud_wr
        ml_ud_wr_master[yr_num].to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                                        'ml_wr_ud_' + str(yr_num) + r'.csv')

    totals_spread.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                         'spread_money_' + 'total' + r'.csv')
    totals_ou.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                     'ou_money_' + 'total' + r'.csv')
    totals_ml_all.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                         'ml_money_all_' + 'total' + r'.csv')
    totals_ml_ud_dol.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                        'ml_money_ud_' + 'total' + r'.csv')
    totals_ml_ud_wr.to_csv(r'models/results/optimal/' + r'/' + str(prev_count) + r'/' +
                        'ml_wr_ud_' + 'total' + r'.csv')
