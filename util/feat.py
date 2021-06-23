import numpy as np
import pandas as pd
import util.dictionaries as di
import util.split as split


############################################### Get X, y #########################################################


def get_X_y_single(prev_count, end_year, stat_names=di.stat_name_simple, start_year=2012):
    """
    Drops predetermined unneeded columns from X which has all game data on a single row.

    :param prev_count: (int)
        The number of previous games we want to use to build our model.
    :param end_year: (int)
        The last season to get data from. Includes end_year's season but not seasons after.
    :param stat_names: list[str]
        stat_name that have been recorded on game day which must be removed
    :param start_year: (int)
        Season to start getting data from, mainly used for test set. Defaults to 2012 when not specified.
    :return: (DataFrame)
        The working X and y dataframes as one. y is the last column(s).
    """

    # Read in the cleaned data and select correct year range
    name = r"data/clean_data_" + str(prev_count) + ".csv"
    data_df = pd.read_csv(name)
    X = data_df.loc[((data_df.season <= end_year) & (data_df.season >= start_year))]
    X.reset_index(drop=True, inplace=True)

    # Remove some columns which will definitely not be features such as team name and recorded stats for the game we
    # are trying to predict. Reorganize columns.
    # Create column indexes as an np.array so that we can select multiple segments of column indexes.
    cols = np.r_[3:10, 16:40]
    X = X.iloc[:, cols]
    X.reset_index(drop=True, inplace=True)

    # Remove other recorded stats for the game other than points as we already explored their correlations and are not
    # trying to predict those.
    temp = [s + '_home' for s in stat_names]
    # temp.remove('pts_home')
    # Also remove season
    # temp.append('season')
    X.drop(temp, inplace=True, axis=1)

    return X


def get_X_y_split(prev_count, end_year, stat_names=di.stat_name_simple, start_year=2012):
    """
    Drops predetermined unneeded columns from X which has all game data on two separate rows.

    :param prev_count: (int)
        The number of previous games we want to use to build our model.
    :param end_year: (int)
        The last season to get data from. Includes end_year's season but not seasons after.
    :param stat_names: list[str]
        stat_name that have been recorded on game day which must be removed
    :param start_year: (int)
        Season to start getting data from, mainly used for test set. Defaults to 2012 when not specified.
    :return: (DataFrame)
        The working X and y dataframes as one. y is the last column(s).
    """

    temp = split.all_split_two_rows(prev_count, end_year, start_year)

    # Remove some columns which will definitely not be features such as team name and recorded stats for the game we
    # are trying to predict. Reorganize columns.
    # Create column indexes as an np.array so that we can select multiple segments of column indexes.
    cols = np.r_[3:10, 16:28, 30:42]
    X = temp.iloc[:, cols]
    X.reset_index(drop=True, inplace=True)

    # Removes pts_home column from X and we re-add it at the end
    temp = X.pop('pts_home')
    X['pts_home'] = temp

    # Remove other recorded stats for the game other than points as we already explored their correlations and are not
    # trying to predict those.
    temp = [s + '_home' for s in stat_names]
    temp.remove('pts_home')
    # Also remove season
    # temp.append('season')
    X.drop(temp, inplace=True, axis=1)

    return X


def get_vegas_y_split(prev_count: int, end_year: int, start_year=2012):
    """
    Returns bookmaker's model's prediction of the response variable.

    :param prev_count: (int)
        The number of previous games we want to use to build our model.
    :param end_year: (int)
        The last season to get data from. Includes end_year's season but not seasons after.
    :param start_year: (int)
        Season to start getting data from, mainly used for test set. Defaults to 2012 when not specified.
    :return: (Series)
        The series containing bookmaker's prediction of response variable.
    """

    temp = split.all_split_two_rows(prev_count, end_year, start_year)
    y = temp.loc[:, 'ou_home']
    y.reset_index(drop=True, inplace=True)
    return y


############################################### Feat Interactions #####################################################


def diff_int(X: pd.DataFrame, type: str):
    if type == '_simple':
        X['win_rate_h'] = X.wins_gotten_h / (X.wins_gotten_h + X.wins_allowed_h)
        X['win_rate_a'] = X.wins_gotten_a / (X.wins_gotten_a + X.wins_allowed_a)
        X['win_rate_diff_to'] = X.win_rate_h - X.win_rate_a
        X.drop(columns=['wins_gotten_h', 'wins_allowed_h', 'wins_gotten_a', 'wins_allowed_a'], inplace=True)

    if type == '_advanced':
        X['pie_diff_to'] = X.pie_gotten_h - X.pie_gotten_a
        X.drop(columns=['pie_gotten_h', 'pie_gotten_a'], inplace=True)

        X['poss_diff_h'] = X.poss_gotten_h - X.poss_allowed_h
        X['poss_diff_a'] = X.poss_gotten_a - X.poss_allowed_a
        X.drop(columns=['poss_gotten_a', 'poss_allowed_h'], inplace=True)

        X['plusminus_diff_to'] = X.plus_minus_gotten_h - X.plus_minus_gotten_a

        X.drop(columns=['net_rating_gotten_h', 'net_rating_gotten_a'], inplace=True)

        X['win_rate_h'] = X.wins_gotten_h / (X.wins_gotten_h + X.wins_allowed_h)
        X['win_rate_a'] = X.wins_gotten_a / (X.wins_gotten_a + X.wins_allowed_a)
        X['win_rate_diff_to'] = X.win_rate_h - X.win_rate_a
        X.drop(columns=['wins_gotten_h', 'wins_allowed_h', 'wins_gotten_a', 'wins_allowed_a'], inplace=True)

        '''X['pts_avg_to1'] = (X.pts_gotten_h*0.52 + X.pts_allowed_a*0.48)
        X.drop(columns=['pts_gotten_h', 'pts_allowed_a'], inplace=True)
        X['pts_avg_to2'] = (X.pts_gotten_a*0.48 + X.pts_allowed_h*0.52)
        X.drop(columns=['pts_gotten_a', 'pts_allowed_h'], inplace=True)'''

        '''X['oreb_pct_avg_to1'] = (X.oreb_pct_gotten_h*0.52 + (1-X.dreb_pct_gotten_a)*0.48)
        X.drop(columns=['oreb_pct_gotten_h', 'dreb_pct_gotten_a'], inplace=True)

        X['oreb_avg_to1'] = (X.oreb_gotten_h * 0.52 + X.oreb_allowed_a * 0.48)
        X.drop(columns=['oreb_gotten_h', 'oreb_allowed_a'], inplace=True)
        X['oreb_avg_to2'] = (X.oreb_gotten_a * 0.48 + X.oreb_allowed_h * 0.52)
        X.drop(columns=['oreb_gotten_a', 'oreb_allowed_h'], inplace=True)

        X['fg3m_avg_to1'] = (X.fg3m_gotten_h * 0.52 + X.fg3m_allowed_a * 0.48)
        X.drop(columns=['fg3m_gotten_h', 'fg3m_allowed_a'], inplace=True)

        X['fg3a_avg_to1'] = (X.fg3a_gotten_h * 0.52 + X.fg3a_allowed_a * 0.48)
        X.drop(columns=['fg3a_gotten_h', 'fg3a_allowed_a'], inplace=True)

        X['fga_avg_to1'] = (X.fga_gotten_h * 0.52 + X.fga_allowed_a * 0.48)
        X.drop(columns=['fga_gotten_h', 'fga_allowed_a'], inplace=True)
        X['fga_avg_to2'] = (X.fga_gotten_a * 0.52 + X.fga_allowed_h * 0.48)
        X.drop(columns=['fga_gotten_a', 'fga_allowed_h'], inplace=True)

        X['ast_avg_to1'] = (X.ast_gotten_h * 0.52 + X.ast_allowed_a * 0.48)
        X.drop(columns=['ast_gotten_h', 'ast_allowed_a'], inplace=True)
        X['ast_avg_to1'] = (X.ast_gotten_a * 0.52 + X.ast_allowed_h * 0.48)
        X.drop(columns=['ast_gotten_a', 'ast_allowed_h'], inplace=True)

        X['fta_avg_to1'] = (X.fta_gotten_h * 0.52 + X.fta_allowed_a * 0.48)
        X.drop(columns=['fta_gotten_h', 'fta_allowed_a'], inplace=True)

        X['poss_avg_to1'] = (X.poss_gotten_h * 0.52 + X.poss_allowed_a * 0.48)
        X.drop(columns=['poss_gotten_h', 'poss_allowed_a'], inplace=True)
        X['poss_avg_to2'] = (X.poss_gotten_a * 0.48 + X.poss_allowed_h * 0.52)
        X.drop(columns=['poss_gotten_a', 'poss_allowed_h'], inplace=True)

        X['efg_pct_to1'] = (X.efg_pct_gotten_h * 0.52 + X.efg_pct_allowed_a * 0.48)
        X.drop(columns=['efg_pct_gotten_h', 'efg_pct_allowed_a'], inplace=True)
        X['efg_pct_to2'] = (X.efg_pct_gotten_a * 0.48 + X.efg_pct_allowed_h * 0.52)
        X.drop(columns=['efg_pct_gotten_a', 'efg_pct_allowed_h'], inplace=True)

        X['off_rating_avg_to1'] = (X.off_rating_gotten_h * 0.52 + X.def_rating_gotten_a * 0.48)
        X.drop(columns=['off_rating_gotten_h', 'def_rating_gotten_a'], inplace=True)'''

    return None
