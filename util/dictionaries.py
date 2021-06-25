from sklearn import linear_model
from sklearn import neural_network
from sklearn import neighbors
from sklearn import ensemble
from sklearn import pipeline
import xgboost as xgb

team_id = {
    1610612737: 'ATL',
    1610612738: 'BOS',
    1610612740: 'NOP',
    1610612741: 'CHI',
    1610612742: 'DAL',
    1610612743: 'DEN',
    1610612745: 'HOU',
    1610612746: 'LAC',
    1610612747: 'LAL',
    1610612748: 'MIA',
    1610612749: 'MIL',
    1610612750: 'MIN',
    1610612751: 'BKN',
    1610612752: 'NYK',
    1610612753: 'ORL',
    1610612754: 'IND',
    1610612755: 'PHI',
    1610612756: 'PHX',
    1610612757: 'POR',
    1610612758: 'SAC',
    1610612759: 'SAS',
    1610612760: 'OKC',
    1610612761: 'TOR',
    1610612762: 'UTA',
    1610612763: 'MEM',
    1610612764: 'WAS',
    1610612765: 'DET',
    1610612766: 'CHA',
    1610612739: 'CLE',
    1610612744: 'GSW',
}

# Some two worded city/team names in the raw odds data have no space separating two words; some do.
# Input both ways to be safe.
team_names = {
    'Atlanta': 'ATL',
    'Boston': 'BOS',
    'NewOrleans': 'NOP',
    'New Orleans': 'NOP',
    'Chicago': 'CHI',
    'Dallas': 'DAL',
    'Denver': 'DEN',
    'Houston': 'HOU',
    'LAClippers': 'LAC',
    'LA Clippers': 'LAC',
    'LALakers': 'LAL',
    'LA Lakers': 'LAL',
    'Miami': 'MIA',
    'Milwaukee': 'MIL',
    'Minnesota': 'MIN',
    'Brooklyn': 'BKN',
    'NewYork': 'NYK',
    'New York': 'NYK',
    'Orlando': 'ORL',
    'Indiana': 'IND',
    'Philadelphia': 'PHI',
    'Phoenix': 'PHX',
    'Portland': 'POR',
    'Sacramento': 'SAC',
    'SanAntonio': 'SAS',
    'San Antonio': 'SAS',
    'OklahomaCity': 'OKC',
    'Oklahoma City': 'OKC',
    'Toronto': 'TOR',
    'Utah': 'UTA',
    'Memphis': 'MEM',
    'Washington': 'WAS',
    'Detroit': 'DET',
    'Charlotte': 'CHA',
    'Cleveland': 'CLE',
    'GoldenState': 'GSW',
    'Golden State': 'GSW',
}

# Counting stats that are scaled via OT minutes.
stat_name_scale = ['pts', 'fgm', 'fga', 'fg3m', 'fg3a', 'ftm', 'fta',
                   'oreb', 'dreb', 'reb', 'ast', 'stl', 'blk', 'tov',
                   'pf', 'poss']

# Game stats wanted as features for _simple version.
# Taken from games_get_avg.py's search for stats with _home
stat_name_simple = ['pts', 'fg_pct', 'ft_pct', 'fg3_pct', 'ast', 'reb']
# Features (the averages of game stats) wanted for _simple version.
# Taken from games_get_avg.py's column names which are used to build 'last_x' DataFrames
feat_name_simple = ['pts_gotten', 'pts_allowed', 'fg_pct_gotten', 'fg_pct_allowed',
                    'ft_pct_gotten', 'ft_pct_allowed', 'fg3_pct_gotten', 'fg3_pct_allowed',
                    'ast_gotten', 'ast_allowed', 'reb_gotten', 'reb_allowed', 'wins_gotten', 'wins_allowed']

# Game stats wanted as features for _advanced version.
stat_name_advanced = ['pts', 'fgm', 'fga', 'fg_pct',
                      'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta',
                      'ft_pct', 'oreb', 'dreb', 'reb', 'ast',
                      'stl', 'blk', 'tov', 'pf', 'plus_minus', 'off_rating',
                      'def_rating', 'net_rating', 'ast_pct', 'ast_tov',
                      'ast_ratio', 'oreb_pct', 'dreb_pct', 'reb_pct',
                      'tm_tov_pct', 'efg_pct', 'ts_pct', 'poss', 'pie', 'wins']
# Features (the averages of game stats) wanted for _advanced version.
# There are also some linearly dependent stats. Such as, reb_pct_gotten will always be 100% - reb_pct_allowed.
# Another example is net_rating_gotten = -net_rating_allowed.
# Thus these two variables are essentially the same and one can be removed.
# Removals = ['off_rating_allowed', 'def_rating_allowed', 'net_rating_allowed', 'plus_minus_away',
#             'oreb_pct_allowed', 'dreb_pct_allowed', 'reb_pct_allowed', 'pie_allowed'])
feat_name_advanced = ['pts_gotten', 'pts_allowed',
                      'fgm_gotten', 'fgm_allowed', 'fga_gotten', 'fga_allowed', 'fg_pct_gotten', 'fg_pct_allowed',
                      'fg3m_gotten', 'fg3m_allowed', 'fg3a_gotten', 'fg3a_allowed', 'fg3_pct_gotten', 'fg3_pct_allowed',
                      'ftm_gotten', 'ftm_allowed', 'fta_gotten', 'fta_allowed', 'ft_pct_gotten', 'ft_pct_allowed',
                      'oreb_gotten', 'oreb_allowed', 'dreb_gotten', 'dreb_allowed', 'reb_gotten', 'reb_allowed',
                      'ast_gotten', 'ast_allowed', 'stl_gotten', 'stl_allowed', 'blk_gotten', 'blk_allowed',
                      'tov_gotten', 'tov_allowed', 'pf_gotten', 'pf_allowed', 'plus_minus_gotten',
                      'off_rating_gotten', 'def_rating_gotten', 'net_rating_gotten',
                      'ast_pct_gotten', 'ast_pct_allowed', 'ast_tov_gotten', 'ast_tov_allowed', 'ast_ratio_gotten',
                      'ast_ratio_allowed',
                      'oreb_pct_gotten', 'dreb_pct_gotten', 'reb_pct_gotten',
                      'tm_tov_pct_gotten', 'tm_tov_pct_allowed', 'efg_pct_gotten', 'efg_pct_allowed',
                      'ts_pct_gotten', 'ts_pct_allowed',
                      'poss_gotten', 'poss_allowed', 'pie_gotten', 'wins_gotten', 'wins_allowed']

# Hyperparam: Previous games used to create average
prev_count = [8, 20, 40, 60, 80]

# Years to run predictions and bet on
test_years = [2015, 2016, 2017, 2018, 2019, 2020]

seed = 12


# Do a quick sample with these first then select one and fine tune
algos_raw = [linear_model.HuberRegressor(fit_intercept=True, max_iter=1000),
             neural_network.MLPRegressor(hidden_layer_sizes=(100,), max_iter=400, batch_size=100,
                                         learning_rate='adaptive', alpha=0.01, random_state=seed),
             neighbors.KNeighborsRegressor(n_neighbors=20, leaf_size=50, weights='uniform', n_jobs=2),
             xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.25, learning_rate=0.1,
                              max_depth=5, reg_lambda=1, reg_alpha=1, n_estimators=500, n_jobs=4,
                              eval_metric='mae', tree_method='gpu_hist', random_state=seed),
             linear_model.Lasso(alpha=.1, fit_intercept=True, max_iter=1000, random_state=seed),
             linear_model.Ridge(alpha=1, fit_intercept=True, max_iter=1000, random_state=seed),
             linear_model.SGDRegressor(loss='epsilon_insensitive', epsilon=0.0001, penalty='elasticnet',
                                       l1_ratio=0.5,
                                       alpha=0.01, learning_rate='adaptive', fit_intercept=True, random_state=seed)
             ]
# Name of algorithms to use for both train and evaluate. Baseline 'Vegas' model must be inserted at the front of list
algos_name = ['HuberRegressor', 'MLPRegressor', 'KNeighborsRegressor',
              'XGBRegressor', 'Lasso', 'Ridge', 'SGDRegressor']

# The difference between our prediction and betting line. Will only place a bet if margin is greater than threshold here
margins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# For optimization of final model, XGBRegressor
margins_opt = [-5, -5.5, -4, -4.5, -3, -3.5, -2, -2.5, -1, -1.5,
               0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
               6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
lower_bounds_opt = [-110, -105, 100, 105, 110, 115, 120]
algos_name_opt = 'XGBRegressor'
num_folds_opt = 5

# For testing of final model
margins_test = [-1.5]
lower_bounds_test = 120
