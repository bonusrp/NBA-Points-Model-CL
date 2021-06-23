import pandas as pd


############################################### Get Bet Info ########################################################


def bet_info(data_df: pd.DataFrame):
    """
    Selects and returns only the needed bet information from a dataframe of cleaned data.

    :return: (DataFrame)
    """
    data_df['ou_total'] = data_df.loc[:, 'ou_home'] + data_df.loc[:, 'ou_away']
    return data_df.loc[:, ['spread', 'ml_home', 'ml_away', 'ou_home', 'ou_away', 'ou_total', 'points_total',
                           'home_margin', 'home_win']]


############################################### Spread #########################################################


# Testing
'''
model_pred = pd.read_csv(r'model_pred.csv')
model_pred = pd.Series(model_pred.iloc[:, 0])
vegas_spread = pd.read_csv(r'vegas_spread.csv')
vegas_spread = pd.Series(vegas_spread.iloc[:, 0])
home_win_margin = pd.read_csv(r'real_spread.csv')
home_win_margin = pd.Series(home_win_margin.iloc[:, 0])
margin = 0
'''


def pts_to_spread(pts_home_df: pd.Series):
    """
    Turns a pts_home series which has been split up into two rows back into one and gets the spread of the points.

    :param pts_home_df: (Series)
    :return: (Series)
    """

    # Force into Series in case a np.array is passed through
    if (type(pts_home_df) != pd.DataFrame) & (type(pts_home_df) != pd.Series):
        pts_home_df = pd.Series(pts_home_df)

    # Get alternating rows of the pts_home series
    df_1 = pts_home_df.iloc[0::2].reset_index(drop=True)
    df_2 = pts_home_df.iloc[1::2].reset_index(drop=True)
    temp = pd.DataFrame()
    temp['pts_home'] = df_1
    temp['pts_away'] = df_2

    # Spread is negative when home team is favoured so I use pts_away - pts_home.
    temp['spread_pred'] = temp.loc[:, 'pts_away'] - temp.loc[:, 'pts_home']
    return temp.spread_pred


def spread_bet(model_spread: pd.Series, vegas_spread: pd.Series, margin):
    """
    Uses predicted spreads and returns a series of bets vs. vegas spread lines. Will also takes a margin input which
    will be a confidence check. Only make bets for which the margin between model_spread and vegas_spread is exceeded.

    :param model_spread: (Series)
        containing predicted model spread.
    :param vegas_spread: (Series)
        containing Vegas's model spread. Betting lines that I are trying to beat.
    :param margin: (int)
        that model_spread must exceed vegas_spread by.
    :return: (Series)
        containing our bets. {-1: 'No bet', 0: 'Bet away spread'; 1: 'Bet home spread'}
    """

    bet_on = vegas_spread.copy(deep=True)

    # Difference between predicted spreads of model and vegas.
    diff = model_spread - vegas_spread
    for diff_index in diff.index:
        temp_diff = diff.iloc[diff_index]
        # If difference is less than 0 it indicates home spread is favoured. Then perform margin check and
        # bet on home spread if passed.
        if temp_diff < 0:
            if abs(temp_diff) >= abs(margin):
                bet_on.iloc[diff_index] = 1
            else:
                bet_on.iloc[diff_index] = -1
        # If difference is greater than 0 it indicates away spread is favoured. Then perform margin check and
        # bet on away spread if passed.
        elif temp_diff > 0:
            if temp_diff >= abs(margin):
                bet_on.iloc[diff_index] = 0
            else:
                bet_on.iloc[diff_index] = -1
        # If difference is 0 its a no bet.
        else:
            bet_on.iloc[diff_index] = -1

    bet_on = bet_on.astype(int)
    return bet_on


def spread_outcome(pts_home_prediction: pd.Series, vegas_spread: pd.Series, home_win_margin: pd.Series, margin=0):
    """
    Evaluates and returns the outcomes of bets on spreads vs. real spreads. Also calculates and returns a win
    percentage and dollars won or lost.

    :param pts_home_prediction: (Series)
        containing points predictions from model
    :param vegas_spread: (Series)
        containing Vegas's model spread or the betting lines.
    :param home_win_margin: (Series)
        containing actual spreads of the games.
    :param margin: (int)
        that model_spread must exceed vegas_spread by.
    :returns:
        (Series) containing bet outcomes. {-1: 'No bet or Push', 0: 'Loss'; 1: 'Win'};
        (float) bet win percentage;
        (int) dollars won or lost from betting $110 each bet.
    """

    # Turn a pts_home series which has been split up into two rows back into one and gets the spread of the points.
    model_spread = pts_to_spread(pts_home_prediction)

    # Use predicted spreads and create a series of bets vs. vegas spread lines. Will also takes a margin input which
    # will be a confidence check.
    bet_on = spread_bet(model_spread, vegas_spread, margin)

    # Turn spread_bet from [0,1] into actual spreads like +1.5
    # To do so just copy vegas_spread and modify
    actual_bet = vegas_spread.copy(deep=True)
    outcome = bet_on.copy(deep=True)
    real_spread = home_win_margin.copy(deep=True)

    # Initially the spreads were set to be from the perspective of home team. However, in spread_bet() I decided
    # which teams to bet on and now can no longer just look at spreads from perspective of home team. So now I must
    # change both actual_bet and real_spread_temp to respect both home/away. It will not be indicated in the
    # dataframe but it will be fixed internally. Get index for bets that will be made on the away spread indicated by 0
    temp_1 = bet_on[bet_on == 0]
    # Since it is away spread it will be the negative of the home spread.
    # E.g. an entry of -5.0 in vegas_spread, initially indicates Home must win by more than 5 pts. Thus, Away must not
    # lose by more than 5. Otherwise written as +5.0 w.r.t Away. This is the 'actual bet' I am making on the game.
    actual_bet.iloc[temp_1.index] = -actual_bet.iloc[temp_1.index]
    # home_win_margin is simply home_margin from data/processed/clean_data.csv and tracks HOME team's margin of victory
    # if positive and margin of loss if negative.
    # E.g. an entry of -5.0 in home_win_margin, initially indicates Home lost by 5 pts. Thus, Away won by 5 points and
    # is +5.0 w.r.t Away.
    # This change is important as it allows us to compare our actual bet to the real spread iff I bet on Away.
    real_spread.iloc[temp_1.index] = -home_win_margin.iloc[temp_1.index]
    # If I bet on Home, all spreads are already in w.r.t home team and so it will be unchanged
    # If bet = -1 it means no bet is placed; simply change all spreads to 0 in this case so bet is assessed as push
    temp_2 = bet_on.loc[bet_on.iloc[:] == -1]
    actual_bet.iloc[temp_2.index] = 0
    real_spread.iloc[temp_2.index] = 0

    # Spreads act as a handicap to a team's score. -5 means a team must win by at least 6 pts, 6-5 = 1 pt, to win
    # the bet. +5 means a team can LOSE by up to 4 pts, -4+5 = 1 pt, and still win the bet.
    # In this way I can calculate a team's new_score with the spread handicap in mind.
    new_score = actual_bet + real_spread

    # If the new_score is positive it means the team I bet on won after the handicap was applied.
    # I win the bet, indicated by 1 in outcomes
    outcome.iloc[(new_score.iloc[:] > 0).loc[(new_score.iloc[:] > 0)].index] = 1
    # If the new_score is negative it means the team I bet on lost after the handicap was applied.
    # I lose the bet, indicated by 0 in outcomes
    outcome.iloc[(new_score.iloc[:] < 0).loc[(new_score.iloc[:] < 0)].index] = 0
    # If the new_score is equal to 0 then the teams tied each other after the handicapped was applied or a bet was not
    # made to begin with. I push the bet, indicated by -1 in outcomes.
    outcome.iloc[(new_score.iloc[:] == 0).loc[(new_score.iloc[:] == 0)].index] = -1

    # Calculate the win percentage
    win = len(outcome.loc[outcome.iloc[:] == 1])
    loss = len(outcome.loc[outcome.iloc[:] == 0])
    # +1 to denominator for stabilisation in case win and loss are both 0
    win_pct = (win / (win + loss + 0.01)) * 100

    # Calculate the amount of money won
    # Odds have a 10% vig factored in so -110. One must bet $110 to win $100.
    dollars = win * 100 - loss * 110

    return outcome, win_pct, dollars


############################################### OverUnder #########################################################


def pts_to_total(pts_home_df: pd.Series):
    """
    Turns a pts_home series which has been split up into two rows back into one row that contains the sum of both
    pts_home, i.e. the total number of points scored in a game.

    :param pts_home_df: (Series)
    :return: (Series)
    """

    # Force into Series in case a np.array is passed through
    if (type(pts_home_df) != pd.DataFrame) & (type(pts_home_df) != pd.Series):
        pts_home_df = pd.Series(pts_home_df)

    # Get alternating rows of the pts_home series
    df_1 = pts_home_df.iloc[0::2].reset_index(drop=True)
    df_2 = pts_home_df.iloc[1::2].reset_index(drop=True)
    temp = pd.DataFrame()
    temp['pts_home'] = df_1
    temp['pts_away'] = df_2

    temp['total'] = temp.loc[:, 'pts_home'] + temp.loc[:, 'pts_away']
    return temp.total


def ou_bet(model_totals: pd.Series, vegas_totals: pd.Series, margin):
    """
    Uses predicted totals and returns a series of bets vs. vegas total lines. Will also takes a margin input which
    will be a confidence check. Only make bets for which the margin between model_totals and vegas_totals is exceeded.

    :param model_totals: (Series)
        containing predicted model totals.
    :param vegas_totals: (Series)
        containing Vegas's model spread. Betting lines that I are trying to beat.
    :param margin: (int)
        that model_totals must exceeded vegas_totals by.
    :return: (Series)
        containing our bets. {-1: 'No bet', 0: 'Bet away spread'; 1: 'Bet home spread'}
    """

    bet_on = vegas_totals.copy(deep=True)

    # Difference between predicted spreads of model and vegas.
    diff = model_totals - vegas_totals
    for diff_index in diff.index:
        temp_diff = diff.iloc[diff_index]
        # If difference is less than 0 it indicates model predicted a lower total than the line. Then perform margin
        # check and bet on under if passed.
        if temp_diff < 0:
            if abs(temp_diff) >= abs(margin):
                bet_on.iloc[diff_index] = 0
            else:
                bet_on.iloc[diff_index] = -1
        # If difference is greater than 0 it indicates model predicted a higher total than the line. Then perform margin
        # check and bet on over if passed.
        elif temp_diff > 0:
            if temp_diff >= abs(margin):
                bet_on.iloc[diff_index] = 1
            else:
                bet_on.iloc[diff_index] = -1
        # If difference is 0 its a no bet.
        else:
            bet_on.iloc[diff_index] = -1

    bet_on = bet_on.astype(int)
    return bet_on


def ou_outcome(pts_home_prediction: pd.Series, vegas_ou: pd.Series, total_pts: pd.Series, margin=0):
    """
    Evaluates and returns the outcomes of bets on over under of combined total points. Also calculates and returns a win
    percentage and dollars won or lost.

    :param pts_home_prediction: (Series)
        containing points predictions from model
    :param vegas_ou: (Series)
        containing Vegas's model for combined point totals, i.e. the betting lines for ou.
    :param total_pts: (Series)
        containing total points scored in a game
    :param margin: (int)
        that model_totals must exceeded vegas_totals by.
    :returns:
        (Series) containing bet outcomes. {-1: 'No bet or Push', 0: 'Loss'; 1: 'Win'};
        (float) bet win percentage;
        (int) dollars won or lost from betting $110 each bet.
    """

    model_total = pts_to_total(pts_home_prediction)
    bet_on = ou_bet(model_total, vegas_ou, margin)
    # Feed actual score totals and vegas's lines into ou_bet to get a ground truth version of which side was the
    # winning bet.
    bet_truth = ou_bet(total_pts, vegas_ou, margin=0)

    # Turn spread_bet from [0,1] into actual spreads like +1.5
    # To do so just copy vegas_spread and modify
    outcome = bet_on.copy(deep=True)

    # Comparing our bets to ground truth bets. Ignore any -1 as we do not bet those. Out of the remaining bets
    # assign a 1 if we match the ground truth, 0 if we do not.
    temp_loss = bet_on[bet_on == 1][bet_on[bet_on == 1] != bet_truth[bet_on[bet_on == 1].index]].index
    outcome[temp_loss] = 0

    # Gets the index of subset df == 0 and compares it to ground truths at the same indices.
    # Then assigns a 1, won bet, if both are 0. Leave it if the two are different as it the row will already be 0
    temp_win = bet_on[bet_on == 0][bet_on[bet_on == 0] == bet_truth[bet_on[bet_on == 0].index]].index
    outcome[temp_win] = 1

    # If ground truth is a -1 it indicates a push and so change the outcome at the same indices to -1.
    # Assign at the end to wipe out any comparisons above that were marked incorrectly cause bet_truth was a -1.
    temp_push = bet_truth[bet_truth == -1].index
    outcome[temp_push] = -1

    # Calculate the win percentage
    win = len(outcome.loc[outcome.iloc[:] == 1])
    loss = len(outcome.loc[outcome.iloc[:] == 0])
    # +1 to denominator for stabilisation in case win and loss are both 0
    win_pct = (win / (win + loss + 0.01)) * 100

    # Calculate the amount of money won
    # Odds have a 10% vig factored in so -110. One must bet $110 to win $100.
    dollars = win * 100 - loss * 110

    return outcome, win_pct, dollars


############################################### Moneyline #########################################################


def ml_bet(model_spread: pd.Series, margin):
    """
    Uses predicted spreads and returns a series of moneyline bets. Will also takes a margin input which
    will be a confidence check.

    :param model_spread: (Series)
        containing predicted model spread.
    :param margin: (int)
        that model_totals must exceeded vegas_totals by.
    :return: (Series)
        containing our bets. {-1: 'No bet', 0: 'Bet away spread'; 1: 'Bet home spread'}
    """

    bet_on = model_spread.copy(deep=True)

    # Assign no bets to spreads that are not greater than our margin threshold
    bet_on[abs(model_spread) < abs(margin)] = -1

    # Create index of spreads which are greater than threshold
    will_bet_index = model_spread[abs(model_spread) > abs(margin)].index
    # Out of those we will bet on, if the spread is negative it indicates home team is favoured. So assign a
    # 1 indicating a moneyline bet on home team.
    temp_home = model_spread[will_bet_index][model_spread[will_bet_index] < 0].index
    bet_on[temp_home] = 1
    # positive spread indicate away team favoured, so assign a 0 to indicate a moneyline bet on away team.
    temp_away = model_spread[will_bet_index][model_spread[will_bet_index] > 0].index
    bet_on[temp_away] = 0

    bet_on = bet_on.astype(int)
    return bet_on


def ml_outcome(pts_home_prediction: pd.Series, vegas_odds: pd.Series, h_win: pd.Series, margin=0, lower_bound_odds=0):
    """
    Evaluates and returns the outcomes of bets on the moneyline. Also calculates and returns a win percentage and
    dollars won or lost.


    :param pts_home_prediction: (Series)
        containing points predictions from model
    :param vegas_odds: (DataFrame)
        containing Vegas's odds for moneyline bets for home and away team in American format.
    :param h_win: (Series)
        containing [0,1] indicating if home team won or lost.
    :param margin: (int)
        that model_totals must exceeded vegas_totals by.
    :param lower_bound_odds: (int)
        that determines the lowest american odds offered that we will still make bets on.
    :returns:
        (Series) containing bet outcomes. {-1: 'No bet or Push', 0: 'Loss'; 1: 'Win'};
        (float) bet win percentage;
        (int) dollars won or lost from betting $110 each bet.
    """

    model_spread = pts_to_spread(pts_home_prediction)
    bet_on = model_spread.copy(deep=True)

    for bet_on_temp in range(len(model_spread)):
        # if home has the nice odds
        if vegas_odds.loc[bet_on_temp, 'ml_home'] >= lower_bound_odds:
            # < -margin here because home spreads are - if favoured
            if model_spread[bet_on_temp] < -margin:
                bet_on[bet_on_temp] = 1
            else:
                bet_on[bet_on_temp] = -1
        # if away has the nice odds
        elif vegas_odds.loc[bet_on_temp, 'ml_away'] >= lower_bound_odds:
            # >+margin here because home spreads are - if favoured
            if model_spread[bet_on_temp] > margin:
                bet_on[bet_on_temp] = 0
            else:
                bet_on[bet_on_temp] = -1
        # else neither odd meets criteria
        else:
            bet_on[bet_on_temp] = -1

    # Turn spread_bet from [0,1] into actual spreads like +1.5
    # To do so just copy vegas_spread and modify
    outcome = bet_on.copy(deep=True)

    # Comparing our bets to ground truth bets. Ignore any -1 as we do not bet those. Out of the remaining bets
    # assign a 1 if we match the ground truth, 0 if we do not.
    temp_loss = bet_on[bet_on == 1][bet_on[bet_on == 1] != h_win[bet_on[bet_on == 1].index]].index
    outcome[temp_loss] = 0

    # Gets the index of subset df == 0 and compares it to ground truths at the same indices.
    # Then assigns a 1, won bet, if both are 0. Leave it if the two are different as it the row will already be 0
    temp_win = bet_on[bet_on == 0][bet_on[bet_on == 0] == h_win[bet_on[bet_on == 0].index]].index
    outcome[temp_win] = 1

    # Calculate the amount of money won/lost
    # Odds for moneyline are assigned rather than -110 so results must treated on a case by case basis
    dollars = 0
    win = 0
    loss = 0

    # Lose $100 per lost bet
    # losses_num = len(outcome[outcome == 0].index)
    # dollars -= losses_num * 100
    loss_index = outcome[outcome == 0].index
    loss_index_h = bet_on[loss_index][bet_on[loss_index] == 1].index
    loss_index_a = bet_on[loss_index][bet_on[loss_index] == 0].index

    for ind in loss_index_h:
        # If odds are > lower_bound and I only bet $100 this means I will $odds
        if vegas_odds.loc[ind, 'ml_home'] >= lower_bound_odds:
            dollars -= 100
            loss += 1
    for ind in loss_index_a:
        if vegas_odds.loc[ind, 'ml_away'] >= lower_bound_odds:
            dollars -= 100
            loss += 1

    # Get subset of all won bets and iterate through them while identifying which one was home and away and then
    # finding the appropriate odds for that team's moneyline bet. Assume 100 dollar bets.
    win_index = outcome[outcome == 1].index
    win_index_h = bet_on[win_index][bet_on[win_index] == 1].index
    win_index_a = bet_on[win_index][bet_on[win_index] == 0].index

    if lower_bound_odds < 0:
        for ind in win_index_h:
            # If odds are < 0, can choose to bet or not bet. If I do bet and I only bet $100, I would
            # only win $[100 / (odds/100)]
            if lower_bound_odds <= vegas_odds.loc[ind, 'ml_home'] < 0:
                # Make sure to turn odds into absolute values.
                dollars += 10000 / abs(vegas_odds.loc[ind, 'ml_home'])
                win += 1
            # If odds are > 0 and I only bet $100 this means I will $odds
            if vegas_odds.loc[ind, 'ml_home'] > 0:
                dollars += vegas_odds.loc[ind, 'ml_home']
                win += 1
        for ind in win_index_a:
            if lower_bound_odds <= vegas_odds.loc[ind, 'ml_away'] < 0:
                dollars += 10000 / abs(vegas_odds.loc[ind, 'ml_away'])
                win += 1
            if vegas_odds.loc[ind, 'ml_away'] > 0:
                dollars += vegas_odds.loc[ind, 'ml_away']
                win += 1
    else:  # When lower bound >= 0
        for ind in win_index_h:
            if vegas_odds.loc[ind, 'ml_home'] >= lower_bound_odds:
                dollars += vegas_odds.loc[ind, 'ml_home']
                win += 1
        for ind in win_index_a:
            if vegas_odds.loc[ind, 'ml_away'] >= lower_bound_odds:
                dollars += vegas_odds.loc[ind, 'ml_away']
                win += 1

    win_pct = (win / (win + loss + 0.01)) * 100
    return outcome, win_pct, dollars


############################################### Multiprocess #########################################################

'''
def bet_outcomes_mp(prediction: pd.Series, bet_df: pd.DataFrame, margins, algos_name_og):
    for threshold in margins:
        # For spread_bets
        spread_outcome, spread_win_pct, spread_dollars = spread_outcome(prediction,
                                                                            bet_df.spread,
                                                                            bet_df.home_margin,
                                                                            margin=threshold
                                                                            )
        # Saves a row of dollars earned for each threshold
        spread.loc[di.algos_name[mod_num], str(threshold)] = spread_dollars

        # For ou_bets
        ou_outcome, ou_win_pct, ou_dollars = bet.ou_outcome(prediction.iloc[:, mod_num],
                                                            bet_df.ou_total,
                                                            bet_df.points_total,
                                                            margin=threshold
                                                            )
        ou.loc[di.algos_name[mod_num], str(threshold)] = ou_dollars
        # For ml_bets
        ml_outcome, ml_win_pct, ml_dollars = bet.ml_outcome(prediction.iloc[:, mod_num],
                                                            bet_df.loc[:, ['ml_home', 'ml_away']],
                                                            bet_df.home_win,
                                                            margin=threshold
                                                            )
        ml.loc[di.algos_name[mod_num], str(threshold)] = ml_dollars
'''
