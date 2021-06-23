import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from util.dictionaries import stat_name_simple
from util import split
from util import feat

import os
from pathlib import Path

# Path(__file__) returns absolute file path. Taking the .parent returns the folder this script is in. The .parent of
# that is the main folder for this project which we wish to use as the main working directory.
os.chdir(Path(__file__).parent.parent)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Set how many previous games we want to use to set the averages
prev_count = 40

# We will take only take seasons prior to 2019. The 2019 and after seasons account for 2168/11361 = 19% of the total
# data. I am assuming this model is being made after the 2017 season and as such I have absolutely no idea about the
# future data. This split is done prior to EDA and feature selection to truly be blind.
end_year = 2020
# I am trying to predict points scored so I will split up each game into two rows. each response variable will the
# points scored by a single team.
temp = split.all_split_two_rows(prev_count, end_year)

# Create and save a boxplot of points differential for each game before removing the relevant column
fig, ax = plt.subplots(figsize=(12, 4))
sns.boxplot(x=abs(temp.home_margin))
plt.axvline(16.0, color='red', linestyle='--', linewidth=2, label='16')
x = plt.xticks(np.arange(0, 65, step=5))
plt.xlabel('Points Differential')
fig.savefig(str(r'eda\graphs\pts_diff.png'))
fig.clf()

# Remove some columns which will definitely not be features such as team name and recorded stats for the game we
# are trying to predict. Reorganize columns.
# Create column indexes as an np.array so that we can select multiple segments of column indexes.
cols = np.r_[3:10, 16:28, 30:42]
X = temp.iloc[:, cols]
X.reset_index(drop=True, inplace=True)
pd.concat([X, X.pts_home], axis=1)
# Removes pts_home column from X and we re-add it at the end
temp = X.pop('pts_home')
X['pts_home'] = temp

# Create and save a heatmap of correlations for each of our 6 main stats and their predictors which are the averages
# for the respective stat.
# Sets plot size for all figures
fig, ax = plt.subplots(figsize=(12, 8))
for stat in stat_name_simple:
    path = r'eda\graphs\corr_' + stat + '.png'
    sns.heatmap(X.loc[:, [str(stat + '_home'), str(stat + '_gotten_h'), str(stat + '_allowed_h'),
                          str(stat + '_gotten_a'), str(stat + '_allowed_a'), 'season']].corr(),
                cmap="Blues", annot=True, fmt='.2f', vmin=0)
    fig.savefig(path)
    fig.clf()

# Remove other recorded stats for the game other than points as we already explored their correlations and are not
# trying to predict those.
temp = [s + '_home' for s in stat_name_simple]
temp.remove('pts_home')
X.drop(temp, inplace=True, axis=1)

# Get a working X for the 8 and [40, 60, 80] game version of the cleaned data
X_1 = feat.get_X_y_split(8, end_year, stat_name_simple)
X_2 = feat.get_X_y_split(20, end_year, stat_name_simple)
temp = pd.concat([X_1.corr()[['pts_home']].transpose(),
                  X_2.corr()[['pts_home']].transpose(),
                  X.corr()[['pts_home']].transpose()])

for prev_count_temp in [60, 80]:
    X_3 = feat.get_X_y_split(prev_count_temp, end_year, stat_name_simple)
    temp = pd.concat([temp, X_3.corr()[['pts_home']].transpose()])
# Merge the two X dataframes with correlations for pts_home.
temp.index = ['pts_home_' + str(8), 'pts_home_' + str(20), 'pts_home_' + str(prev_count),
              'pts_home_' + str(60), 'pts_home_' + str(80)]
# Omit pts_home, the last column, as it has a 1:1 correlation to itself
temp.drop('pts_home', axis=1, inplace=True)
# Create and save a heatmap for just correlation of explanatory variables and response variable for both 20 and 80 games
fig, ax = plt.subplots(figsize=(14, 4))
sns.set_context('paper', font_scale=1.05)
sns.heatmap(temp, cmap="RdPu", annot=True, fmt='.2f', vmin=0)
plt.subplots_adjust(top=0.68, bottom=0.42)
plt.tick_params(labelsize=9)
plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, va="center")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=75)
fig.savefig(str(r'eda\graphs\corr_prev_count.png'))
fig.clf()

# Correlation heat map for all variables. needs larger size to fit all the extra rows/columns
fig, ax = plt.subplots(figsize=(16, 12.8))
sns.heatmap(X.corr(), cmap="RdPu", annot=True, fmt='.2f', vmin=0)
fig.savefig(str(r'eda\graphs\corr_all_' + str(prev_count) + '.png'))
fig.clf()

# TODO: rename subplot row axis to not include the _a
# Creates a plot with 14 subplots
fig, ax = plt.subplots(7, 2, figsize=(13, 16), sharey=False)
# Manually adjust hspace between each plot as x_label was being cut off
plt.subplots_adjust(left=0.05, bottom=0.05, wspace=.14, hspace=.40)
# Treats ax, subplot positions, as a vector rather an array
ax = ax.flatten()
# Create histograms with KDE helping us visualize the distributions of our variables
# Do not plot seasons which is in the first column so start index at 1
for i in range(13, len(X.columns)):
    # For pts_home create the histogram with binwidth 1 to better visualize the data.
    if X.columns[i].startswith('pts_h'):
        # ax[i-1] because we are start with index of 1 on X.columns but we want to plot to index 0 on subplot
        sns.histplot(x=X[X.columns[i]], kde=True, ax=ax[i-13], binwidth=1,
                     binrange=[min(X.loc[:, X.columns[i]]), max(X.loc[:, X.columns[i]])])
        # Remove the _h or _a on variable names cause we group the pair. If its target variable "pts_home"
        # do not remove anything.
        ax[i - 13].set(xlabel=X.columns[i])
    else:
        sns.histplot(x=X[X.columns[i]], kde=True, ax=ax[i-13])
        # Remove the _h or _a on variable names cause we group the pair.
        ax[i-13].set(xlabel=X.columns[i][:-2])
fig.savefig(str(r'eda\graphs\hist_kde.png'))
fig.clf()
