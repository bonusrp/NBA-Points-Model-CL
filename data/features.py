import pandas as pd
import numpy as np
from tqdm import tqdm
from util import split
from util import feat
from util import dictionaries as di

# TODO: Do the corr variable kill thing here. @games_get_avg.py's todo
# or we can just not and for the really egregious one which is poss_all/_got we can get the 'difference' of them.
# but this is phase three after we get another baseline with _advanced data model.
'''
corr_matrix = feats.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features
df.drop(to_drop, axis=1, inplace=True)'''
################### Use this one to decide which one to keep by checking the pair with the corr
# with pts_home and keeping higher one
'''# Loop through columns to find any row/col that has high correlation.
# keep list of the two
for column in upper.columns:
    temp = upper.loc[upper.loc[:, column] > 0.95]
    if not temp.empty:
        print('this is row,', temp.index.values)
        print('this is col,', column)
'''

# For each data set create a csv which only includes features (averages), target variable (pts_), and
# bookmaker's prediction of the target variable (ou_).
data_name = [r"processed\clean_data_simple_", r"processed\clean_data_advanced_"]
data_type = ["_simple", "_advanced"]
feat_file_name = [r"features\feat_simple_", r"features\feat_advanced_"]
prev_counts = di.prev_count.copy()
for i in tqdm(range(len(data_name))):
    f_names = "feat_name" + data_type[i]
    f_names = getattr(di, f_names)
    df = []
    df_names = []

    for prev_count in tqdm(prev_counts, colour='#00ff00'):
        data_df = pd.read_csv(data_name[i] + str(prev_count) + '.csv')

        feats = split.feat_split_two_rows(data_df, f_names)
        # TODO: potentially expand on this section with the feat interactions and how we will go about choosing
        #  em. Can do it for only a few high correlation ones, can try to learn it for every prediction which
        #  will be next level compute. Can just set 0.52/0.48 arbitrary for all just for testing and if its better
        #  on median error AND betting results THEN we can do learning every single stat.
        # Create interaction variables
        feat.diff_int(feats, data_type[i])
        # Append the features df
        df.append(feats)
        # Append what will be the filename for this feature df
        df_names.append(feat_file_name[i] + str(prev_count) + '.csv')

    # Dropping too highly correlated ones
    temp = []
    for j in range(len(df)):
        temp_df = df[j].loc[:, df[j].columns != 'pts_home']
        temp_df = df[j].loc[:, df[j].columns != 'ou_home']
        corr_matrix = temp_df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        temp.append(to_drop)
    # Need to get the names of the longest list in order to remove columns uniformly across multiple prev_counts
    temp_len = [len(sublist) for sublist in temp]
    to_drop1 = temp[temp_len.index(max(temp_len))]

    # Dropping low corr ones
    temp = pd.DataFrame()
    for j in range(len(df)):
        temp_df = df[j].loc[:, df[j].columns != 'ou_home']
        # Get the correlation of each feature with dependent variable pts_home for each df of different prev_counts
        # Omit ou_home, the last column, from feats df for this calculation
        temp = pd.concat([temp, temp_df.corr().abs()[['pts_home']].transpose()])

    # Omit pts_home, the last column, as it has a 1:1 correlation to itself
    # If ALL corr in a column are < 0.05 add the col name to a list
    to_drop2 = [column for column in temp.columns[:-1] if all(temp[column] < 0.05)]

    # Get only the unique names in both to_drop lists
    to_drop = np.unique(np.array(to_drop1 + to_drop2))
    # Cannot remove season since it will be used to split training/test sets.
    if 'season' in to_drop:
        to_drop = to_drop[to_drop != 'season']
    # Keep this as it wsa 'incorrectly' dropped. If we drop each variable one by one it wont be dropped meaning w/e it
    # was correlated with was already removed and thus win_rate_diff does not need ot be removed.
    if 'win_rate_diff_to' in to_drop:
        to_drop = to_drop[to_drop != 'win_rate_diff_to']
    print('Dropped:', to_drop)

    for j in range(len(df)):
        # Drop columns using the list of names of low corr variables then save it
        df[j].drop(to_drop, axis=1, inplace=True)
        df[j].to_csv(df_names[j], index=False)
