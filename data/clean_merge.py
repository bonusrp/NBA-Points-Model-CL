import pandas as pd
import sys
from tqdm import tqdm
from util import dictionaries as di

games_name = [r"interim\clean_games_simple_", r"interim\clean_games_advanced_"]
odds_name = [r"interim\clean_odds_with_post.csv", r"interim\clean_odds_with_post.csv"]
data_name = [r"processed\clean_data_simple_", r"processed\clean_data_advanced_"]
prev_counts = di.prev_count.copy()
for i in tqdm(range(len(games_name))):
    for prev_count in prev_counts:
        # Compare and the dates from cleaned games and cleaned odds as a precursor to merging the two dataframes
        name = games_name[i] + str(prev_count) + ".csv"
        games_df = pd.read_csv(name)
        odds_df = pd.read_csv(odds_name[i])
        check = (games_df.game_date == odds_df.date)
        if all(~check):
            print(check[~check])
            sys.exit("Dates do not match.")
        # Using the above check, I discovered that there was a mismatch caused by 5 Finals games of the 2019-20 season
        # which were played in October due to Covid causing a temporary shutdown. This was the only time the second
        # half of a season was played in a double digit month causing one of my conditional blocks to mislabel the date.
        # Another game was misdated from the source. We will manually fix these errors in clean_odds.py

        # Sort both dataframes using game date first and home team name second
        games_df.sort_values(by=["game_date", 'team_id_home'], ascending=[True, True], inplace=True)
        games_df.reset_index(drop=True, inplace=True)
        odds_df.sort_values(by=["date", 'home'], ascending=[True, True], inplace=True)
        odds_df.reset_index(drop=True, inplace=True)
        # Another check. If both checks are passed without error then the games are ordered in the same way
        # in both datasets
        check = (games_df.team_id_home == odds_df.home)
        if all(~check):
            print(check[~check])
            sys.exit("Home team names do not match.")

        # Merge games and odds then clean redundant columns
        data_df = pd.concat([games_df, odds_df], axis=1)
        data_df.drop(['date', 'home', 'away'], axis=1, inplace=True)
        data_df.columns.values[0:3] = ['date', 'home', 'away']

        # Adding a game outcome indicator column. 1 = home team win, 0 = home team loss.
        data_df = data_df.assign(home_win=data_df['home_margin'])
        data_df.loc[data_df['home_margin'] >= 0, 'home_win'] = 1
        data_df.loc[data_df['home_margin'] < 0, 'home_win'] = 0

        # Saving merged data
        data_df.columns = data_df.columns.str.lower()
        data_df.home_win = data_df.home_win.values.astype(bool)
        print('\nSAVED')
        name = data_name[i] + str(prev_count) + '.csv'
        data_df.to_csv(name, index=False)
