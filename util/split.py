import pandas as pd
import os
import numpy as np
from sklearn import preprocessing

############################################### Split Games #####################################################



def feat_split_two_rows(data_df: pd.DataFrame, feat_name: list):
    """
    FOR FEATURES AND TARGET VARIABLE ONLY. Splits each game into two consecutive rows. Switches the home and away stats
    such that every team is treated as the home team in their respective row.

    :param data_df: DataFrame to split games from
    :param feat_name: List
    :return: DataFrame with split games
    """
    # Using feat_name to build a list of all relevant names for home/away
    feat_home = pd.Series(feat_name) + '_h'
    feat_away = pd.Series(feat_name) + '_a'
    keep = list(feat_home)
    # extend adds elements to list
    keep.extend(list(feat_away))
    # Will also need season and target variable, pts. Also betting line for pts, ou_
    keep.extend(['season', 'pts_home', 'pts_away', 'ou_home', 'ou_away'])

    data_df = data_df.loc[:, keep]

    # Since I am predicting points scored so I will split up each game into two rows.
    # I will treat the team who's points scored are being predicted as the Home team.
    # Duplicate all rows and reorder using index
    temp = pd.concat([data_df] * 2).sort_index()
    # Must reset_index here because I will be using .loc where index matters while all_split_two_rows uses .iloc
    # where index doesn't matter.
    temp.reset_index(drop=True, inplace=True)

    # Insert column to indicate whether the team we are predicting is actually the home or away team.
    # This will be a feature but can also help us when looking at the dataset since column names will treat every
    # team as the 'home team' and their opponents as 'away team' regardless of their actual status.
    # Will insert it right before 'season' column so will need to find the index of that col.
    temp.insert(int(np.where(temp.columns.values == 'season')[0]), 'is_home', 0)
    # For the actual home teams change is_home from 0 to 1
    temp.loc[0::2, 'is_home'] = 1

    # For away teams move their stats, denoted by _a, to _h columns
    temp.loc[1::2, feat_home] = temp.loc[1::2, feat_away].values
    temp.loc[1::2, feat_away] = temp.loc[0::2, feat_home].values
    # Swap pts_home and _away
    temp.loc[1::2, 'pts_home'] = temp.loc[1::2, 'pts_away'].values
    # Swap pts_home and _away
    temp.loc[1::2, 'ou_home'] = temp.loc[1::2, 'ou_away'].values
    # Drop unneeded away stats
    temp.drop(columns=['pts_away', 'ou_away'], inplace=True)

    return temp


def all_split_two_rows(prev_count, end_year, start_year=2012):
    """
    FOR ALL DATA IN _SIMPLE ONLY. Splits each game into two consecutive rows. Switches the home and away stats
    such that every team is treated as the home team in their respective row.

    :param prev_count: (int)
        The number of previous games we want to use to build our model.
    :param end_year: (int)
        The last season to get data from. Includes end_year's season but not seasons after.
    :param start_year: (int)
        Season to start getting data from, mainly used for test set. Defaults to 2012 when not specified.
    :return: (DataFrame)
        The updated X dataframe
    """
    path = os.getcwd() + r'\data\processed'

    # Read in the cleaned data and select correct year range
    name = path + r"\clean_data_simple_" + str(prev_count) + ".csv"
    data_df = pd.read_csv(name)
    X = data_df.loc[((data_df.season <= end_year) & (data_df.season >= start_year))]
    X.drop('home_win', axis=1, inplace=True)
    X.reset_index(drop=True, inplace=True)

    # I am trying to predict points scored so I will split up each game into two rows.
    # I will treat the team who's points scored are being predicted as the Home team.
    # Duplicate all rows and reorder using index
    temp = pd.concat([X] * 2).sort_index()

    # For away teams move their stats, denoted by _away, to _home columns
    # 1::2 lets us get every other row starting from i=1
    temp.iloc[1::2, 4:10] = temp.iloc[1::2, 10:16].values
    # Swap ou_home and ou_away
    temp.iloc[1::2, -4] = temp.iloc[1::2, -3].values
    # Swap _a and _h for average stats, for ml, for spread
    temp.iloc[1::2, 16:28] = temp.iloc[1::2, 30:42].values
    temp.iloc[1::2, 30:42] = temp.iloc[0::2, 16:28].values
    # Swap ml
    temp.iloc[1::2, -6] = temp.iloc[1::2, -5].values
    # Swap spread
    temp.iloc[1::2, -7] = -temp.iloc[1::2, -7].values

    return temp


################################################## Non-Split #########################################################


def feat_one_row(data_df: pd.DataFrame, feat_name: list, target_var: str):
    """
    FOR FEATURES AND TARGET VARIABLE ONLY. Keeps each game as one row and returns a df of features and target variable.


    :param data_df: DataFrame to split games from
    :param feat_name: List
    :param target_var: String of name of target variable to keep
    :return: DataFrame with split games
    """
    # Using feat_name to build a list of all relevant names for home/away
    feat_home = pd.Series(feat_name) + '_h'
    feat_away = pd.Series(feat_name) + '_a'
    keep = list(feat_home)
    # extend adds elements to list
    keep.extend(list(feat_away))
    # Will also need season and target variable, pts. Also betting line for pts, ou_
    keep.extend(['season', target_var])

    data_df = data_df.loc[:, keep]

    return data_df
