import os
import pandas as pd
from tqdm import tqdm
from util.dictionaries import team_names


# Data for a single game is stored in 2 back to back rows. If the point spread value in 'Close' is on row 2,
# it indicates that team_2, which is home team, is favoured by that amount of points. e.g. team 1 +5, team 2 -5.
# The other value in row 1 of 'Close' indicates the O/U for points total.

# Specify directory housing raw odds data
directory = 'odds'

df_new = pd.DataFrame(columns=['Date', 'Home', 'Away', 'OU', 'Spread', 'ML_Home', 'ML_Away', 'OU_Home', 'OU_Away',
                               'Points_total', 'Home_Margin'])

# Loop through each file in the odds directory
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    # Getting the season year from xlsx file name
    season = filename[9:13]

    if filename.endswith('.xlsx'):
        # Initializing our storage dataframe
        df = pd.read_excel(directory + '\\' + filename)
        x = pd.DataFrame(columns=['Date', 'Home', 'Away', 'OU', 'Spread', 'ML_Home', 'ML_Away', 'OU_Home', 'OU_Away',
                                  'Points_total', 'Home_Margin'])
        # Use a count to help us know which row # of 2 I am on
        count = 2
        # itertuples should be faster than using a loop and .loc but can test later
        for row in df.itertuples():
            # First row is visiting team
            if count % 2 == 0:
                # Due to the way NBA seasons work, if date is 4 length this signifies it has a double digit month which
                # means it was played in the first year of a season. e.g. in 2010 of 2010-11 season.
                if len(str(row[1])) == 4:
                    # YYYY-MM-DD format to match our data\games.csv format for dates
                    date = season + '-' + str(row[1])[0:2] + '-' + str(row[1])[2:]
                # Otherwise it is assumed the game was played in the second year of a season.
                else:
                    date = str(int(season) + 1) + '-' + '0' + str(row[1])[0:1] + '-' + str(row[1])[1:]
                # Uses our dictionary to turn city name into
                away = team_names.get(str(row[4]))
                # 'pk' in the 'Close' column indicates a spread of 0. Assume anything not a float is treated as 'pk'.
                if type(row[11]) == str:
                    ou = 0
                else:
                    ou = row[11]
                ml_away = str(row[12])
                points = row[9]
                count += 1
            # Second row is home team
            else:
                home = team_names.get(str(row[4]))
                if type(row[11]) == str:
                    spread = 0
                else:
                    spread = row[11]

                # Because O/U and spread are mixed up and both under 'Close' I need a way to differentiate them.
                # The row that the spread is on indicates that the team on that row is favoured. I will be referencing
                # the spread w.r.t the HOME team. E.g. -5 means HOME team is favoured by 5 points. I can identify which
                # number in 'Close' is the spread and which is O/U by their size. Spread will always be less than the
                # O/U since for a team to win by x points they will have to at least score x points. Thus, expected
                # points total will be at least x. If our 'spread' var is greater than our 'ou' var we will switch them.
                if spread > ou:
                    temp = spread
                    spread = ou
                    ou = temp
                # If spread is correctly placed in the second row for HOME team we will take the negative of it to
                # indicate they are favoured.
                else:
                    spread = -spread
                ml_home = str(row[12])

                # Calculate predicted bookmaker's prediction for how many points home and away team will score
                # Since we do not have the actual data we can estimate it via formula: OU total/2 +- the spread/2.
                ou_home = (ou/2) - (spread/2)
                ou_away = (ou/2) + (spread/2)

                # Calculations for margin of victory and actual point total used to assess bet outcomes.
                margin = row[9] - points
                points += row[9]
                temp = {
                    'Date': date, 'Home': home, 'Away': away, 'OU': ou, 'Spread': spread,
                    'ML_Home': ml_home, 'ML_Away': ml_away, 'OU_Home': ou_home, 'OU_Away': ou_away,
                    'Points_total': points, 'Home_Margin': margin
                }
                x = x.append(temp, ignore_index=True)
                count += 1

        # Append cleaned odds at the end of each xlsx to our larger dataframe housing cleaned odds for all seasons
        df_new = df_new.append(x, ignore_index=True)

# Fix 6 specific games having mismatched dates. Errors found using clean_merge.py
df_new.loc[10331:10335, 'Date'] = df_new.loc[10331:10335, 'Date'].str.replace('2019', '2020')
df_new.at[10374, 'Date'] = df_new.at[10374, 'Date'].replace('28', '27')
# Fix one game with missing closing spread. Will use opening spread instead.
df_new.at[9611, 'Spread'] = -5.5
df_new.at[9611, 'OU_Home'] = (df_new.at[9611, 'OU']/2) - (df_new.at[9611, 'Spread']/2)
df_new.at[9611, 'OU_Away'] = (df_new.at[9611, 'OU']/2) + (df_new.at[9611, 'Spread']/2)

# Sort by newest game first and store into csv
df_new.sort_values(by=["Date"], ascending=False, inplace=True)
df_new.columns = df_new.columns.str.lower()

# Print .info to see if there are any missing values and that data types are correct.
# If there are any nulls in any cell then print the rows with missing values and do not save the file.
if any(df_new.isnull().any()):
    df_new.info()
    print(df_new[df_new.isnull().any(axis=1)])
    print('NOT SAVED')
else:
    df_new.info()
    print('SAVED WITH POSTSEASON')
    name = r'interim\clean_odds_with_post.csv'
    df_new.to_csv(name, index=False)

    # Remove post season games using a list of post season start and end dates for each season
    # 2019 season includes 1 west play-in game, so playoffs technically started on 2020-08-15.
    # Same for every season going forward.
    postseason_start = ['2013-04-20', '2014-04-19', '2015-04-18', '2016-04-16', '2017-04-15', '2018-04-14',
                        '2019-04-13', '2020-08-15', '2021-05-18']
    postseason_end = ['2013-06-20', '2014-06-15',    '2015-06-16', '2016-06-19', '2017-06-12', '2018-06-08',
                      '2019-06-13', '2020-10-11', '2021-06-18']
    for i in range(len(postseason_start)):
        df_new = df_new.loc[
            ~((df_new.loc[:, 'date'] >= postseason_start[i]) & (df_new.loc[:, 'date'] <= postseason_end[i]))
        ]
    print('SAVED WITHOUT POSTSEASON')
    name = r'interim\clean_odds_no_post.csv'
    df_new.to_csv(name, index=False)
