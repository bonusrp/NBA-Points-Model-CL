import pandas as pd
from util.dictionaries import stat_name_scale

# Merge all box scores. Then save to csv and read again so that duplicated columns have format 'name.1'
games_trad = pd.read_csv(r"raw/games_trad.csv")
games_adv = pd.read_csv(r"raw/games_adv.csv")
data_df = pd.concat([games_trad, games_adv], axis=1)
# Doing this because there are two columns, both called 'MIN', tracking minutes but one is inaccurate
# for total team minutes.
data_df.to_csv(r'interim/games_box.csv', index=False)
data_df = pd.read_csv(r"interim/games_box.csv")

# Check for na values
# data_df.info()

# make all col names lower cases
data_df.columns = data_df.columns.str.lower()
# Remove duplicate and redundant columns. E.g. team's ids, names, city names, and abbreviations and their duplicates.
# Just need one column of TEAM_IDs that will be mapped to team abbrev.
# Remove estimated advanced stats which are built using a team's estimated possessions. Keep advanced stats calculated
# using actual possessions. However, there is evidence that
# https://squared2020.com/2017/07/10/analyzing-nba-possession-models/ even the NBA's 'actual' possession stats
# are still slightly inaccurate.
data_df.drop(columns=[
    'team_abbreviation',
    'team_name',
    'wl',
    'min',
    'game_id.1',
    'team_id.1',
    'team_name.1',
    'team_abbreviation.1',
    'team_city',
    'e_off_rating',
    'e_def_rating',
    'e_net_rating',
    'e_tm_tov_pct',
    'e_usg_pct',
    'usg_pct',
    'e_pace',
    'pace',
    'pace_per40',
], inplace=True)

# Attempt to clean new games data to make it in the same form as original games data so that code can be reused
# Change season_id to season. season_id's last 4 digits is the season year.
data_df.loc[:, 'season_id'] = data_df.loc[:, 'season_id'].astype(str)
data_df.loc[:, 'season_id'] = data_df.loc[:, 'season_id'].str[-4:]
data_df.rename(columns={'season_id': 'season'}, inplace=True)

# Due to some games going to OT (even up to four OTs) some games have heavily inflated stats. We will use MIN.1 which
# tracks how many minutes a team actually played and scale each of our counting stats by this number. This is not
# completely accurate because it will scale the stats recorded in OT as well which may not be representative of
# normal conditions due to fatigue but still it remains a decent estimate. Note: this will not effect spreads and
# spread bets as those will still be graded on the final score which is the rule most bookmakers operate by.
# Scale counting stats using. A regular game is 240:00 team minutes played = 1.
data_df['scale'] = data_df.loc[:, 'min.1'].str[:3].astype(int) / 240
data_df.loc[:, stat_name_scale] = data_df.loc[:, stat_name_scale].divide(data_df.loc[:, 'scale'], axis=0)
data_df.drop(columns=['min.1', 'scale'], inplace=True)

# Every two rows contains the data for each team for their specific match but the order of
# home/away team is inconsistent. Will make it consistent home in first row, away in second.
# Grab matchup for first rows of each game. If it contains a vs. is in the match up string then it indicates home
# team is on the first row. If @ is in the match up string then away team is on top.
temp = data_df.loc[::2, 'matchup'].loc[data_df.loc[::2, 'matchup'].str.contains('@')]
# We increase the index of the misordered away teams by 1 and decrease index of misordered home teams similarly.
away = data_df.iloc[temp.index, :].set_index(temp.index + 1)
home = data_df.iloc[temp.index + 1, :].set_index(temp.index)
# Drop misordered rows and append properly ordered rows
data_df.drop(temp.index, axis=0, inplace=True)
data_df.drop(temp.index + 1, axis=0, inplace=True)
data_df = pd.concat([data_df, home, away], axis=0, ignore_index=False)
data_df.sort_index(inplace=True)
data_df.drop(columns='matchup', inplace=True)

# Merging the home and away team into one row. Will differentiate stats using _home for home, _away for away.
# Not exactly necessary but it will make it easier to reuse clean_merge.py
# Split every other row into two dataframes
home = data_df.iloc[0::2, :]
# Will not append _home, _away to shared stats such as game_id
stat_name_shared = ['season', 'game_id', 'game_date']
temp = home.columns[~home.columns.isin(stat_name_shared)].values
temp += '_home'
# Create a dictionary, using zip, which maps our new names to subset of old col names
home.rename(columns=dict(zip(home.columns[~home.columns.isin(stat_name_shared)].values, temp)), inplace=True)
# Reset index so home and away row index matches
home.reset_index(drop=True, inplace=True)

# Do the same for away teams
away = data_df.iloc[1::2, :]
stat_name_shared = ['season', 'game_id', 'game_date']
temp = away.columns[~away.columns.isin(stat_name_shared)].values
temp += '_away'
away.rename(columns=dict(zip(away.columns[~away.columns.isin(stat_name_shared)].values, temp)), inplace=True)
# Drop columns that have shared info as home dataframe will already contain it
away.drop(columns=stat_name_shared, inplace=True)
away.reset_index(drop=True, inplace=True)

# Concatenate the info
data_df = pd.concat([home, away], axis=1)
# Pop and reinsert to rearrange columns
temp = data_df.pop('team_id_away')
data_df.insert(2, temp.name, temp)
temp = data_df.pop('game_date')
data_df.insert(0, temp.name, temp)
temp = data_df.pop('season')
data_df.insert(3, temp.name, temp)
data_df.drop(columns='game_id', inplace=True)

# There are NA values for oreb_pct, dred_pct, reb_pct. Luckily the formulas for these stats are available online
# making it a simple calculation to fill in the missing data.
# Upon further review NBA's calculations for all reb_pct seems to be a bit off.
# Possibly caused by them calculating each player's individual reb_pct then weighing them by minutes played and
# calculating the team's reb_pct off that. Formula is straight forward and basketball reference's calculations line
# up with my own so we will just recalculate all rows.
data_df.loc[:, 'oreb_pct_home'] = data_df.loc[:, 'oreb_home'].divide(data_df.loc[:, 'oreb_home'] +
                                                                     data_df.loc[:, 'dreb_away'])
data_df.loc[:, 'oreb_pct_away'] = data_df.loc[:, 'oreb_away'].divide(data_df.loc[:, 'oreb_away'] +
                                                                     data_df.loc[:, 'dreb_home'])
data_df.loc[:, 'dreb_pct_home'] = data_df.loc[:, 'dreb_home'].divide(data_df.loc[:, 'dreb_home'] +
                                                                     data_df.loc[:, 'oreb_away'])
data_df.loc[:, 'dreb_pct_away'] = data_df.loc[:, 'dreb_away'].divide(data_df.loc[:, 'dreb_away'] +
                                                                     data_df.loc[:, 'oreb_home'])
data_df.loc[:, 'reb_pct_home'] = data_df.loc[:, 'reb_home'].divide(data_df.loc[:, 'reb_home'] +
                                                                   data_df.loc[:, 'reb_away'])
data_df.loc[:, 'reb_pct_away'] = data_df.loc[:, 'reb_away'].divide(data_df.loc[:, 'reb_home'] +
                                                                   data_df.loc[:, 'reb_away'])

# Check for na values
data_df.info()
data_df.to_csv(r'interim/games_box.csv', index=False)

# Code to append playoffs and non_playoffs
'''
b = pd.read_csv(r'data/raw/games_trad_reg.csv')
c = pd.read_csv(r'data/raw/games_trad_playoffs.csv')
a = b.append(c, ignore_index=True)
temp1 = a.iloc[27542:27544]
temp2 = a.iloc[:27490]
a = temp2.append(temp1, ignore_index=True)
a.to_csv(r'data/raw/games_trad.csv', index=False)

b = pd.read_csv(r'data/raw/games_adv_reg.csv')
c = pd.read_csv(r'data/raw/games_adv_playoffs.csv')
a = b.append(c, ignore_index=True)
temp1 = a.iloc[27542:27544]
temp2 = a.iloc[:27490]
a = temp2.append(temp1, ignore_index=True)
a.to_csv(r'data/raw/games_adv.csv', index=False)
'''

# Code to append new data to simple model if needed and also remove data that i dont have odds for
'''
b = pd.read_csv(r'data/interim/games_box.csv')
d = pd.read_csv(r'data/raw/games.csv')
b.sort_values(by=["game_date"], ascending=False, inplace=True)

b = b.iloc[:27500]

# b = b.loc[b.SEASON_ID != 42020]
# b = b.iloc[:27552]
# c = c.loc[c.SEASON_ID != 42020]
# c = c.iloc[:27498]
# b.to_csv(r'data/raw/games_trad.csv')
# c.to_csv(r'data/raw/games_adv.csv')
'''

# b = pd.read_csv(r'data/interim/games_box.csv')
# c = pd.read_csv(r'data/raw/games.csv')
# temp = b.loc[:, ['game_date', 'team_id_home', 'team_id_away', 'season', 'pts_home', 'fgm_home', 'fga_home']]
