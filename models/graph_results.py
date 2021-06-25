import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import util.dictionaries as di

import os
from pathlib import Path

if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.options.mode.chained_assignment = None

    ################################################## Phase 1, 2 #####################################################

    model_results = [r"models/results/simple/80/", r"models/results/advanced/80/"]
    model_type = ["simple", "advanced"]
    test_years_str = list(map(str, di.test_years))

    for i in range(len(model_results)):
        # Read in mae and explained_variance data without the last column 'Total'
        mae_df = pd.read_csv(model_results[i] + 'mae.csv').iloc[:, :-1]
        exvar_df = pd.read_csv(model_results[i] + 'ex_var.csv').iloc[:, :-1]

        # Creating graph of simple, 80, models' mae and exvar.
        # Pivot metrics values from each year into one column
        temp1 = pd.DataFrame(np.vstack([mae_df.iloc[:, [0, 1]]]), columns=['model', 'mae'])
        temp2 = pd.DataFrame(np.vstack([exvar_df.iloc[:, [0, 1]]]), columns=['model', 'explained_variance'])
        for yr_ind in range(1, len(test_years_str)):
            temp1 = pd.DataFrame(np.vstack([temp1, mae_df.iloc[:, [0, yr_ind + 1]]]), columns=['model', 'mae'])
            temp2 = pd.DataFrame(np.vstack([temp2, exvar_df.iloc[:, [0, yr_ind + 1]]]),
                                 columns=['model', 'explained_var'])
        mae_df = temp1.copy()
        exvar_df = temp2.copy()

        # Creates a plot with 2 subplots
        fig, ax = plt.subplots(1, 2, figsize=(18, 10), sharey='col')
        plt.subplots_adjust(left=0.05, bottom=0.15, wspace=.123, hspace=.25)
        ax = ax.flatten()
        sns.barplot(x='model', y='mae', data=mae_df, capsize=.2, palette="husl", ax=ax[0])
        sns.barplot(x=exvar_df.model, y=exvar_df.explained_var * 100, capsize=.2, palette="husl", ax=ax[1])
        plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=22, fontsize=11)
        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=22, fontsize=11)
        ax[0].set_ylabel('Mean Absolute Error (pts)', fontsize='large')
        ax[1].set_ylabel('Explained Variance (%)', fontsize='large')
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        ax[0].axhline(np.mean(mae_df.loc[mae_df.model == 'Vegas'].mae), color='red', linestyle='--',
                      linewidth=2)
        ax[1].axhline(np.mean(exvar_df.loc[exvar_df.model == 'Vegas'].explained_var) * 100, color='red', linestyle='--',
                      linewidth=2)
        if i == 1:
            ax[1].set_ylim([-5, 35])
        else:
            ax[1].set_ylim([0, 35])
        fig.suptitle(r"Various Models' Performance Over All Test Years", fontsize='xx-large')
        fig.savefig(str('models/graphs/' + model_type[i] + r'/model_comparison_' + model_type[i] + '_80.png'))
        fig.clf()

        # Creating graph of model metrics vs. model dollars earned
        # Only taking the mean error across all test years for each model. col = -1
        # Will also not need vegas's model as it cant make bets. row != 0
        metrics_list = [pd.read_csv(model_results[i] + 'mae.csv').iloc[1:, -1].reset_index(drop=True),
                        pd.read_csv(model_results[i] + 'rmse.csv').iloc[1:, -1].reset_index(drop=True),
                        pd.read_csv(model_results[i] + 'med.csv').iloc[1:, -1].reset_index(drop=True),
                        pd.read_csv(model_results[i] + 'ex_var.csv').iloc[1:, -1].reset_index(drop=True) * 100]
        metrics_list_names = ['MEAN_AE', 'RMSE', 'MED_AE', 'EXPLAINED_VAR_PCT']
        # Take for threshold = 1 only
        spread_tot = pd.read_csv(model_results[i] + 'spread_money_total.csv').iloc[:, [2]]
        ou_tot = pd.read_csv(model_results[i] + 'ou_money_total.csv').iloc[:, [2]]
        ml_tot = pd.read_csv(model_results[i] + 'ml_money_all_total.csv').iloc[:, [2]]

        # Concatenating and stacking the relevant information
        for m_ind in range(len(metrics_list)):
            metrics_list[m_ind] = pd.DataFrame(np.vstack([pd.concat([metrics_list[m_ind], spread_tot], axis=1),
                                                          pd.concat([metrics_list[m_ind], ou_tot], axis=1),
                                                          pd.concat([metrics_list[m_ind], ml_tot], axis=1)]),
                                               columns=['metric_measure', 'net_dollars'])
        # Stacking errors of all types and then contacting a column to identify which error is which
        metrics_dollars = pd.DataFrame(np.vstack(metrics_list), columns=['metric_measure', 'net_dollars'])
        metrics_dollars['metric_type'] = 'null'
        temp_count = 0
        for row_ind in range(0, len(metrics_dollars), int(len(metrics_dollars) / len(metrics_list))):
            metrics_dollars.loc[row_ind:row_ind + int(len(metrics_dollars) / len(metrics_list)), 'metric_type'] = \
                metrics_list_names[temp_count]
            temp_count += 1

        # Create scatter plot
        temp = sns.lmplot(data=metrics_dollars, x="net_dollars", y="metric_measure", hue="metric_type",
                          palette="husl", ci=75, truncate=False, height=7, aspect=1.143)

        # Need to use the commands for FacetGrids found in sns docu
        temp.legend.set_title('Metric Type')  # Get legend object which functions as it does in matplotlib and rename
        temp.set_ylabels('Metric Measure')
        temp.set_xlabels('Net Gain ($)')

        # Need to get the .fig of the figure temp because it is actually a facet grid and so we must get the underlying
        temp.fig.subplots_adjust(top=.95)
        temp.fig.suptitle(r"Various Performance Metrics vs. Dollars Earned from Betting (Threshold=1)")
        temp.savefig(str('models/graphs/' + model_type[i] + r'/metrics_dollars_' + model_type[i] + '_80.png'))
        temp.fig.clf()

    ################################################## Optimization ###################################################

    opt_mlud = pd.read_csv(r'models/optuna/advanced/80/opt_mlud.csv')

    z = opt_mlud.lower_odds.unique()
    col = [sns.color_palette("husl", 8)[0], sns.color_palette("husl", 8)[5],
           sns.color_palette("husl", 8)[7], '#4A46CE', '#851e3e', '#D71616', '#7bc043']
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Remove white space around plot for better fit into latex
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=.12, hspace=.25)
    for lower_temp in range(len(z)):
        fig = sns.regplot(x=opt_mlud.threshold[opt_mlud.loc[opt_mlud.loc[:, 'lower_odds'] == z[lower_temp]].index],
                          y=round(opt_mlud.dollars[opt_mlud.loc[opt_mlud.loc[:, 'lower_odds'] == z[lower_temp]].index],
                                  0),
                          y_jitter=100, color=col[lower_temp], line_kws={"lw": 1.3},
                          order=3, ci=None, label=str(z[lower_temp]))

    fig.set_ylim([min(min(opt_mlud.dollars) - 3000, -5000), max(opt_mlud.dollars) + 3000])
    fig.set_xlim([min(opt_mlud.threshold) - 0.5, max(opt_mlud.threshold) + 0.5])

    fig.set_xticks(range(int(min(opt_mlud.threshold)), int(max(opt_mlud.threshold)) + 1, 1))

    fig.legend(title='Lowest Odds')
    fig.set_ylabel('Net Gain ($)')
    fig.set_xlabel('Bet Difference Threshold')

    fig.set_title(r"Underdog Moneyline Hyperparameter Optimization")
    fig.figure.savefig(str('models/graphs/optimal/mlud_opt_advanced_80.png'))
    fig.figure.clf()

    ################################################## Final Model ####################################################

    # Read in mae and explained_variance data without the last column 'Total'
    mae_df = pd.read_csv(r'models/results/optimal/80/mae.csv', index_col=0).iloc[:, :-1]
    exvar_df = pd.read_csv(r'models/results/optimal/80/ex_var.csv', index_col=0).iloc[:, :-1]

    temp1 = mae_df.stack().reset_index()
    temp1.columns = ['model', 'year', 'mae']
    temp2 = exvar_df.stack().reset_index()
    temp2.columns = ['model', 'year', 'exvar']
    mae_df = temp1.copy()
    exvar_df = temp2.copy()
    # Change 'year' from string into int for easier graph coordinate manipulation
    mae_df.loc[:, 'year'] = mae_df.loc[:, 'year'].astype('int32')
    exvar_df.loc[:, 'year'] = exvar_df.loc[:, 'year'].astype('int32')
    exvar_df.loc[:, 'exvar'] = exvar_df.loc[:, 'exvar'] * 100

    # Creating custom colour palette
    col = [sns.color_palette("husl", 8)[6], '#7bc043']
    custom_pal = sns.color_palette(col)

    # Creates a plot with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharex='row')
    plt.subplots_adjust(left=0.05, bottom=0.15, wspace=.12, hspace=.25)
    ax = ax.flatten()
    # Use marker here (NO 's') since markers must be used with style, e.g. style=1, which makes
    # another entry on the legend
    sns.lineplot(data=mae_df, x='year', y='mae', hue="model", palette=custom_pal,
                 marker='o', dashes=False, ax=ax[0])
    sns.lineplot(x=exvar_df.year, y=exvar_df.exvar, hue=exvar_df.model, palette=custom_pal,
                 marker='o', dashes=False, ax=ax[1])

    mod_type_temp = ['Vegas', 'XGB']
    for mod_type_num in range(len(mod_type_temp)):  # Label points on the plot first subplot
        for x, y in zip(mae_df.loc[mae_df.loc[:, 'model'] == mod_type_temp[mod_type_num]].year,
                        mae_df.loc[mae_df.loc[:, 'model'] == mod_type_temp[mod_type_num]].mae):
            # the position of the data label relative to the data point can be adjusted by adding/subtracting a value
            # from the x &/ y coordinates
            ax[0].text(x=x - 0.1,  # x-coordinate position of data label
                       y=y - 0.235,  # y-coordinate position of data label, adjusted to be 150 below the data point
                       s='{: .2f}'.format(y),  # data label, formatted to ignore decimals
                       color=col[mod_type_num])  # set colour of line
    for mod_type_num in range(len(mod_type_temp)):  # Second subplot
        for x, y in zip(exvar_df.loc[exvar_df.loc[:, 'model'] == mod_type_temp[mod_type_num]].year,
                        exvar_df.loc[exvar_df.loc[:, 'model'] == mod_type_temp[mod_type_num]].exvar):
            # the position of the data label relative to the data point can be adjusted by adding/subtracting a value
            # from the x &/ y coordinates
            ax[1].text(x=x - 0.16,  # x-coordinate position of data label
                       y=y - 1.2,  # y-coordinate position of data label, adjusted to be 150 below the data point
                       s='{: .1f}'.format(y),  # data label, formatted to ignore decimals
                       color=col[mod_type_num])  # set colour of line

    for ax_num in [0, 1]:
        plt.setp(ax[ax_num].xaxis.get_majorticklabels(), rotation=22, fontsize=11)
        ax[ax_num].set_xlabel('')

        if ax_num == 0:
            ax[ax_num].set_ylabel('Mean Absolute Error (pts)', fontsize='large')
            ax[ax_num].set_ylim([0, 10])
        else:
            ax[ax_num].set_ylabel('Explained Variance (%)', fontsize='large')
            ax[ax_num].set_ylim([0, 30])

    # Create a shared legend
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].get_legend().remove()
    ax[1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.17, 1), title='Models')

    fig.suptitle(r"Yearly Model Performance", fontsize='xx-large')
    fig.savefig(str('models/graphs/optimal/model_comparison_final_80.png'))
    fig.clf()

    # year-by-year graph for ud ml
    mlud_df = pd.DataFrame()
    for yr_ind in range(len(di.test_years)):
        temp_df = pd.read_csv(r'models/results/optimal/80/ml_money_ud_' + str(yr_ind) + '.csv', index_col=0)
        # After iterating through the separate year csv, create a dataframe of years and dollars earned that year
        mlud_df = mlud_df.append(pd.DataFrame([di.test_years[yr_ind], int(temp_df.iloc[:, 0].values)]).T,
                                 ignore_index=True)

        temp_df = pd.read_csv(r'models/results/optimal/80/ml_wr_ud_' + str(yr_ind) + '.csv', index_col=0)
        # Create a dataframe of years and win percentage
        mlud_df.loc[yr_ind, 'win_pct'] = temp_df.iloc[0].values
    mlud_df.columns = ['year', 'dollars', 'win_pct']
    mlud_df.loc[:, 'year'] = mlud_df.loc[:, 'year'].astype('int32')

    '''
    # Stacked plots
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # creates the second y axis.
    # plots the first set of data, and sets it to ax1.
    sns.barplot(x=mlud_df.year.astype('str'), y=mlud_df.dollars, ax=ax1, color=sns.color_palette("husl", 2)[0])
    # plots the second set, and sets to ax2.
    sns.lineplot(x=mlud_df.year.astype('str'), y=mlud_df.win_pct, marker='o', color=sns.color_palette("husl", 2)[1],
                 linewidth=3.5, ax=ax2)
    '''

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    plt.subplots_adjust(left=0.05, bottom=0.15, wspace=.12, hspace=.25)
    ax = ax.flatten()

    sns.barplot(x=mlud_df.year, y=mlud_df.dollars, color=col[0], ax=ax[0])
    sns.lineplot(x=mlud_df.year, y=mlud_df.win_pct, marker='o', color=col[1],
                 ax=ax[1])

    # Need different method to get coordinates for barplots
    for p in ax[0].patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + p.get_height()
        if y >= 0:
            ax[0].text(x=x, y=y + 75, s='$' + str(int(y)),
                       color='black', ha='center')
        else:
            ax[0].text(x=x, y=y - 200, s='$' + str(int(y)),
                       color='black', ha='center')
    for x, y in zip(mlud_df.year, mlud_df.win_pct):
        # Manual positioning
        if x == 2018 or x == 2015 or x == 2019 or x == 2017:
            ax[1].text(x=x - 0.12, y=y - 0.13, s='{: .1f}%'.format(y), color='black')
        elif x == 2020:
            ax[1].text(x=x - 0.17, y=y + 0.05, s='{: .1f}%'.format(y), color='black')
        else:
            ax[1].text(x=x - 0.12, y=y + 0.05, s='{: .1f}%'.format(y), color='black')

    for ax_num in [0, 1]:
        ax[ax_num].set_xlabel('')

        if ax_num == 0:
            ax[ax_num].set_ylabel('Net Gain ($)', fontsize='large')
        else:
            ax[ax_num].set_ylabel('Win Rate (%)', fontsize='large')

    fig.suptitle(r"Yearly Betting Performance ($100 per bet)", fontsize='xx-large')
    fig.savefig(str('models/graphs/optimal/betting_perf_final_80.png'))
    fig.clf()
