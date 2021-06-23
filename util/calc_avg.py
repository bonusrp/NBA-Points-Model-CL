import numpy as np
import pandas as pd

############################################### Averages #########################################################
pd.set_option("display.max_rows", None, "display.max_columns", None)


def match_stats(games_df_3, games_df_earlier_3, team_id_3, prev_count_3, start_date_3, end_date_3):
    """
    Function which returns the specified number of previous game data for a team. Starts from a specified date which
    is the game date of the 'current' game we wish to predict. Grabs the team's home and away games indiscriminately.

    :param games_df: (DataFrame)
        The dataframe containing all the relevant data of all games we wish to analyze.
    :param games_df_earlier: (DataFrame)
        The dataframe containing all the relevant data from games prior to the ones we wish to analyze. Required so we
        can get data for the oldest games in our prediction subset.
    :param team_id: (int)
        The number representing the team who we wish to grab data for.
    :param prev_count: (int)
        The number of previous games we want to get.
    :param start_date: (str)
        The date of the game we want to predict.
    :param end_date: (str)
        The end date of the range we wish to grab previous games from.
    :return: match_stats: (DataFrame)
        The dataframe containing the correct number of previous game data.
    """

    # Tested using timeit and it's faster to create one new df with overlapping condition check vs. creating two df
    # to not overlap the date range check.
    matches = games_df_3.loc[
        (games_df_3.game_date > end_date_3) & (games_df_3.game_date < start_date_3) &
        ((games_df_3.team_id_home == team_id_3) | (games_df_3.team_id_away == team_id_3))
        ]
    game_count = len(matches)

    # If previous games played < prev_count, go to the 2010/2011 seasons to get the data from there
    if matches.empty or game_count < prev_count_3:
        # Push end date back even further to account for 2011 lockout
        end_date_3 = np.datetime64(end_date_3)
        end_date_3 -= 250
        end_date_3 = str(end_date_3)
        last_season = games_df_earlier_3.loc[
            (games_df_earlier_3.game_date > end_date_3) & (games_df_earlier_3.game_date < start_date_3) &
            ((games_df_earlier_3.team_id_home == team_id_3) | (games_df_earlier_3.team_id_away == team_id_3))
            ]
        matches = matches.append(last_season, ignore_index=True)
    if len(matches) < prev_count_3:
        print(start_date_3)
        print(end_date_3)
        print(team_id_3)
        print(len(matches))

    # Cutting down to only keep most recent prev_count number of games. (Could put a check here to take prev_count
    # if we got. If we don't then just take w/e we have. But for now this will break if we don't have prev_count
    # number of games)
    return matches.iloc[:prev_count_3].reset_index(drop=True)


def store_last_x(matches_4, team_id_4, last_x_4):
    """
    Stores the desired stats for the team's previous x number of games based on whether specified team is the home
    team or away team.

    Stats: pts, fg_pct, ft_pct, fg3_pct, ast, reb.

    :param matches: (DataFrame)
        The dataframe containing previous games data.
    :param team_id: (int)
        The number representing the team who we wish to grab stats for.
    :param last_x: (DataFrame)
        The empty dataframe for storage of our desired stats for each game.
    :return: (DataFrame)
        The updated dataframe last_x.
    """
    # If our team is home team we take their own stats from _home and store it into our last_x stats_team.
    home = matches_4.loc[matches_4.loc[:, 'team_id_home'] == team_id_4].copy()
    # If our team is away team we take their own stats from _away and store it into our last_x stats_team.
    away = matches_4.loc[matches_4.loc[:, 'team_id_home'] != team_id_4].copy()

    last_x_4_temp = last_x_4.copy()

    # Calculate the team's wins gotten and allowed in this subset of games for the team
    wins_gotten = len(home.loc[home.loc[:, 'home_win'] == 1])
    wins_gotten += len(away.loc[away.loc[:, 'home_win'] == 0])
    wins_allowed = len(matches_4) - wins_gotten

    # Add the info to our two home/away match sets. will just call both sides 'wins_h' or 'wins_a' even though it
    # only tracks information for the team in our team_id.
    home.drop('home_win', axis=1, inplace=True)
    # Insert after all the other _home stats for the home subset. Take 4 columns out as those are shared. Then take
    # half as stats have both _home/_away. This gives the length of _home stats. Add 4 back to get the last column
    # of home stats.
    home.insert(4 + int((len(home.columns) - 4) / 2), 'wins_home', wins_gotten)
    home.insert(len(home.columns), 'wins_away', wins_allowed)

    away.drop('home_win', axis=1, inplace=True)
    away.insert(4 + int((len(away.columns) - 4) / 2), 'wins_home', wins_allowed)
    away.insert(len(away.columns), 'wins_away', wins_gotten)

    # Start at 4 or 4+int(len(last_x.columns)/2) because first 4 columns will always be game info and not stats.
    # As long as data was cleaned properly.
    last_x_4_temp.iloc[home.index, 0:len(last_x_4.columns):2] = home.iloc[:, 4:4 + int(len(last_x_4.columns) / 2)]
    # If our team is home team we take their opponent's stats from _away and store it into
    # last_x stats_opp (our team's against stats).
    # Start index at 1 this time for last_x because we want _opp stats only
    last_x_4_temp.iloc[home.index, 1:len(last_x_4.columns):2] = home.iloc[:, 4 + int(len(last_x_4.columns) / 2):]

    last_x_4_temp.iloc[away.index, 1:len(last_x_4.columns):2] = away.iloc[:, 4:4 + int(len(last_x_4.columns) / 2)]
    # If our team is away team we take their opponent's stats from _home and store it into
    # last_x stats_opp (our team's against stats).
    last_x_4_temp.iloc[away.index, 0:len(last_x_4.columns):2] = away.iloc[:, 4 + int(len(last_x_4.columns) / 2):]

    return last_x_4_temp


def get_avg_row(df_new_row, prev_2, last_x_H_2, last_x_A_2, games_df_2, games_df_earlier_2):
    """
    Takes a row (Series) of one game's data and returns the same row with both team's averages appended. Was originally
    in box_get_avg.py in a loop, but now moved here to allow parallel processing.
    """
    # Change date to datetime in order to perform arithmetic
    start_date = np.datetime64(df_new_row.loc['game_date'])
    # Because I am only taking prev 20 games I want to set an end date of games to store to limit operation time.
    # Even assuming 1 whole lockout/shutdown season (365D) + a sparse schedule prior (80*3=240D)
    # I should find at least 80 games in the past 605 days
    end_date = start_date - (565 + prev_2 * 3)
    # Turn our dates back into strings for comparison purposes
    start_date = str(start_date)
    end_date = str(end_date)
    team_H = df_new_row.loc['team_id_home']
    team_A = df_new_row.loc['team_id_away']

    # Getting previous games data in date range for this game's HOME team
    match_H = match_stats(games_df_3=games_df_2, games_df_earlier_3=games_df_earlier_2, team_id_3=team_H,
                          prev_count_3=prev_2, start_date_3=start_date, end_date_3=end_date)
    # Storing specific stats I desire into last_x_* dataframe for further analysis
    last_x_H_2_temp = store_last_x(matches_4=match_H, team_id_4=team_H, last_x_4=last_x_H_2)
    # Calculating averages and storing them. Use 4+stats_count as the first 4 is shared stats. stats+count is the
    # subset following which contains the true values for the game that day, both _home and _away. Then the
    # next stats_count subset that we wish to use as storage contains stats averages
    # for home team, both _team and _opp.
    df_new_row.loc[last_x_H_2_temp.columns + '_h'] = last_x_H_2_temp.mean().values

    # Same for away team
    match_A = match_stats(games_df_3=games_df_2, games_df_earlier_3=games_df_earlier_2, team_id_3=team_A,
                          prev_count_3=prev_2, start_date_3=start_date, end_date_3=end_date)
    last_x_A_2_temp = store_last_x(matches_4=match_A, team_id_4=team_A, last_x_4=last_x_A_2)
    df_new_row.loc[last_x_A_2_temp.columns + '_a'] = last_x_A_2_temp.mean().values

    return df_new_row


# helpers function basically takes each chunk of dataframe and runs pd.DataFrame.apply on them.
def get_avg_df(df, prev_1, last_x_H_1, last_x_A_1, games_df_1, games_df_earlier_1):
    """
    Takes a DataFrame of games data and returns the same df with both team's averages appended. Uses pd.df.apply to
    iterate row by row and feed it into get_avg_row.
    """
    # Lambda iterates through rows of df and passes those rows along with other arguments into the composite function,
    # get_avg_row.
    df_news = df.apply(lambda x: get_avg_row(df_new_row=x, prev_2=prev_1, last_x_H_2=last_x_H_1, last_x_A_2=last_x_A_1,
                                             games_df_2=games_df_1, games_df_earlier_2=games_df_earlier_1), axis=1)
    return df_news
