import pandas as pd
import numpy as np

from util import calc_avg as hp
from util.dictionaries import team_id
from util.dictionaries import prev_count

from timeit import default_timer as timer
from multiprocessing import Pool

if __name__ == '__main__':
    start = timer()
    # Data reading and cleaning
    # Read in the game data which is stored as csv
    games_df = pd.read_csv(r"raw\games.csv")
    # Change to and store a html that is easier to read than a csv
    # games_df.to_html('data/games_df.html')

    # Removing games prior to 2008 season which was when the last team change occurred. Seattle -> OKC
    # But they count as the same team in this dataset so ignore this
    # games_df = games_df.loc[games_df.season >= 2008]

    # Making all col name lower case
    games_df.columns = games_df.columns.str.lower()
    games_df.rename(columns={'game_date_est': 'game_date'}, inplace=True)

    # Regular season games are denoted with ID = 2xxxxxxx. Payoff games ID = 4xxxxxxx. Preseason games ID = 1xxxxxxx.
    # Preseason games generally have a different flow and bench players plays majority of the minutes.
    # Thus I will remove those games as they may not accurately predict 'real' games.
    games_df = games_df.loc[(games_df.game_id >= 20000000)]
    # Some games were inputted twice for 2020 season. I remove any duplicated seasons which are supposed to be unique.
    games_df.drop_duplicates(subset='game_id', keep='first', inplace=True)

    # Sorting by most recent game date first
    games_df.sort_values(by=["game_date"], ascending=False, inplace=True)
    # Updating index and dropping the storage of old index values
    games_df.reset_index(drop=True, inplace=True)

    # Add a column tracking home_wins
    conditions = [
        (games_df.loc[:, 'pts_home'] > games_df.loc[:, 'pts_away']),
        (games_df.loc[:, 'pts_home'] < games_df.loc[:, 'pts_away'])
    ]
    values = [1, 0]
    games_df['home_win'] = np.select(conditions, values)

    # Subset df which will store previous games in 2010 and 11 seasons for when I need earlier than 2012 data
    games_df_earlier = games_df.loc[(games_df.season >= 2010) & (games_df.season < 2012)]
    games_df_earlier.reset_index(drop=True, inplace=True)

    # Removing entries before 2012 season for our working subset
    games_df = games_df.loc[games_df.season >= 2012]

    # Removing unwanted features
    features_games = ["game_date", "team_id_home", "team_id_away", "season", "pts_home", "fg_pct_home",
                      "ft_pct_home", "fg3_pct_home", "ast_home", "reb_home", "pts_away", "fg_pct_away",
                      "ft_pct_away", "fg3_pct_away", "ast_away", "reb_away", 'home_win']
    games_df_earlier = games_df_earlier[features_games].copy(deep=True)
    games_df = games_df[features_games].copy(deep=True)

    # Name of stats to track. Simply all stats with _home (or _away) appended to the end except for team_id_home.
    # Using a DataFrame to use the .insert method later.
    col_name = pd.DataFrame(games_df.columns[games_df.columns.str.contains('_home')][1:])
    # Save a version of this with just stat name for later function call
    stat_name = col_name.iloc[:, 0].copy(deep=True)
    col_name.columns = ['name']
    # Removing the _home portion
    col_name.iloc[:, 0] = col_name.iloc[:, 0].str[:-5]
    # Creating a Series of name [stat1_gotten, stat1_allowed,...] for every stat.
    # Increment index by 2 because we will be inserting a new _allowed row right after every original row.
    stats_count = len(col_name) * 2
    col_name = col_name.T
    for i in range(0, stats_count, 2):
        # Insert a column that has str(i)_o as column name but the real value we care about is the value the column
        # contains, which is a string, statname_allowed
        col_name.insert(i + 1, column=str(i) + '_o', value=col_name.iloc[0, i] + '_allowed')
        # The value in the original column will be statname_gotten
        col_name.iloc[0, i] = col_name.iloc[0, i] + '_gotten'
    col_name = col_name.T
    col_name = col_name.append(pd.Series({'name': 'wins_gotten'}), ignore_index=True)
    col_name = col_name.append(pd.Series({'name': 'wins_allowed'}), ignore_index=True)
    col_name.reset_index(drop=True, inplace=True)

    # I decided to use 8, 20, 40, 60, 80 previous games to get averages.
    for prev in prev_count:
        # Initializing dataframes for looping
        # Creating a zeros np.float64 df which stores the stats (stats_count columns) for each game (prev_count rows)
        # FOR HOME TEAM
        # Numpy creates the empty array which I force into df using Pandas
        last_x_H = pd.DataFrame(np.zeros((prev, len(col_name))))
        # pts_gotten - points per game the team scores or 'gets'
        # pts_allowed - points per game the team allows
        # fg_pct - field goal percent
        # ft_pct - free throw percent
        # fg3_pct - three point field goal percent
        # ast - assists per game
        # reb - rebounds per game
        last_x_H.columns = col_name.iloc[:, 0]
        # Do the same for away team by creating a copy of the df for HOME team and renaming columns
        last_x_A = last_x_H.copy(deep=True)

        # Creating a copy of our games_df which will I will use to fill in the stats
        games_df_new = games_df.copy(deep=True)
        # Getting current # of columns as I will rename new columns that I append
        temp = len(games_df_new.columns)
        # Perform appending via columns. Do not ignore_index as I need the column name
        games_df_new = pd.concat([games_df_new, last_x_H], ignore_index=False, axis=1)
        # Manipulate the values of the index (as the index is immutable) by adding a '_H' to the end
        # signifying home team
        games_df_new.columns.values[temp:] += '_h'
        # Same thing for away team
        temp = len(games_df_new.columns)
        games_df_new = pd.concat([games_df_new, last_x_A], ignore_index=False, axis=1)
        games_df_new.columns.values[temp:] += '_a'

        # MULTIPROCESS 244 (2 splits, pool 3?)
        # MULTIPROCESS 55 (6 splits, pool 8) look to bring this down a bit takes a bit too much ram
        # MULTIPROCESS 76 (4 splits, pool 4) seems about right
        # from 350s to 60s
        # Splits df into x number of sub df stored in list
        df_split = np.array_split(games_df_new, 6)
        pool = Pool(6)
        # Use starmap as it takes a list of iterable tuples that are unpacked as arguments for the function.
        temp = pd.concat(pool.starmap(hp.get_avg_df, [
            (df_split[0], prev, last_x_H, last_x_A, games_df, games_df_earlier),
            (df_split[1], prev, last_x_H, last_x_A, games_df, games_df_earlier),
            (df_split[2], prev, last_x_H, last_x_A, games_df, games_df_earlier),
            (df_split[3], prev, last_x_H, last_x_A, games_df, games_df_earlier),
            (df_split[4], prev, last_x_H, last_x_A, games_df, games_df_earlier),
            (df_split[5], prev, last_x_H, last_x_A, games_df, games_df_earlier)
        ]))
        pool.close()
        pool.join()
        games_df_new = temp.copy(deep=True)

        # Replace team ids with the abbreviated form of team name
        games_df_new.replace({'team_id_home': team_id}, inplace=True)
        games_df_new.replace({'team_id_away': team_id}, inplace=True)

        # Print .info to see if there are any missing values and that data types are correct.
        # If there are any nulls in any cell then print the rows with missing values and do not save the file.
        if any(games_df_new.isnull().any()):
            games_df_new.info()
            print(games_df_new[games_df_new.isnull().any(axis=1)])
            print('NOT SAVED')
        else:
            games_df_new.info()
            print('SAVED ' + str(prev))
            name = r"interim\clean_games_simple_" + str(prev) + ".csv"
            games_df_new.to_csv(name, index=False)

    end = timer()
    print(end - start)