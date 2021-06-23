import pandas as pd
from time import sleep

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoreadvancedv2

# Automatically building season strings from a range of years. Each season starts in one year and ends in the next.
seasons_list = pd.Series(dtype=str)
yr = range(10, 22, 1)
for i in yr:
    seasons_list.loc[i] = '20' + str(i) + '-' + str(i + 1)
seasons_list.reset_index(drop=True, inplace=True)

# Getting a df of regular season games season by season
games_trad = pd.DataFrame()
for s in seasons_list:
    # league_id indicates just nba games
    game_finder = leaguegamefinder.LeagueGameFinder(league_id_nullable='00',
                                                    season_nullable=str(s))
    games_dict = game_finder.get_normalized_dict()
    games = games_dict['LeagueGameFinderResults']
    games = pd.DataFrame(games)
    # Since season list may go beyond current season check that there are actually results in games to not throw error
    if len(games.columns) > 0:
        # Remove game_ids beginning with 1 as they are preseason games
        games = games.loc[~games.GAME_ID.str.startswith('001')]
        # Remove game_ids beginning with 3 as they are all star games
        games = games.loc[~games.GAME_ID.str.startswith('003')]
        # Keeping only 002-regular_season, 005-play_ins, 004-playoffs
        games.sort_values('GAME_ID', ascending=True, inplace=True)
        games.reset_index(drop=True, inplace=True)
        games_trad = games_trad.append(games)
    sleep(0.5)
# Save to csv
games_trad.to_csv(r'raw/games_trad.csv', index=False)
print(len(games_trad))
print('saved')

# Set up variables needed to grab advanced box scores using game ids
# games_adv = pd.read_csv(r'games_adv_3.csv')
games_adv = pd.DataFrame()
temp = pd.read_csv(r"raw/games_trad.csv")
id_list = temp.loc[:, 'GAME_ID']

# Due to nba occasionally blocking api requests, set up a loop to save data every 1000 games in case something breaks.
start_index = range(0, len(id_list), 1000)
count = 1
for i in start_index:
    # Since LeagueGameFinder returned two lines one for each team's traditional stats per game, there are
    # duplicated game ids.
    for ids in id_list[i:i + start_index.step:2].values:
        # Reg season nba game ids start with '00' but when saved to csv the 00 is removed. So manually re-add '00'
        # Also need to specify team box scores or it will return stats per player.
        games_adv = games_adv.append(
            boxscoreadvancedv2.BoxScoreAdvancedV2(game_id='00' + str(ids)).team_stats.get_data_frame()
        )
        games_adv.reset_index(drop=True, inplace=True)
        sleep(0.5)
        print(ids)
    # Save to a incremental backup system
    games_adv.to_csv(r'raw/games_adv_' + str(count) + '.csv', index=False)
    count += 1

# Sort box scores to have matching game ids and team ids and row indexes
games_trad.sort_values(by=["GAME_ID", 'TEAM_ID'], ascending=[True, True], inplace=True)
games_trad.reset_index(drop=True, inplace=True)
games_adv.sort_values(by=["GAME_ID", 'TEAM_ID'], ascending=[True, True], inplace=True)
games_adv.reset_index(drop=True, inplace=True)

games_adv.to_csv(r'raw/games_adv.csv', index=False)
games_trad.to_csv(r'raw/games_trad.csv', index=False)
