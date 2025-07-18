import numpy as np
import pandas as pd

def get_battery_psc(splines, E_b):
    """ 
    Get p_sc as the min of all splines evaluated on [E_b, 1]

    :param splines: np.array with shape (n_splines, 2)
    :param E_b: float with current battery state of charge

    :return p_sc: estimated short-circuit power
    """
    features = np.array([E_b, 1] * splines.shape[0]).reshape(-1, 2)
    assert splines.shape == features.shape

    return min(np.diag(splines @ features.T))

def solve_for_pi(p_sc, p_b):
    """ 
    Solve battery efficiency quadratic equation: p_i**2 - p_sc * p_i + p_sc * p_b
    
    :param p_sc: float with short-circuit power
    :parram p_b: float with external battery power

    :return pi: internal battery power
    """
    sol_pos = (p_sc + np.sqrt(p_sc**2 - 4 * p_b * p_sc)) / 2
    sol_neg = (p_sc - np.sqrt(p_sc**2 - 4 * p_b * p_sc)) / 2

    sol_arr = np.array([sol_pos, sol_neg])

    return sol_arr[np.min(np.abs(sol_arr)) == np.abs(sol_arr)][0]

def get_battery_pi(p_b, E_b, splines):
    """ 
    Estimate internal battery power
    
    :parram p_b: float with external battery power
    :param E_b: float with current battery state of charge
    :param splines: np.array with shape (n_splines, 2)

    :return pi: internal battery power
    """
    p_sc_estimate = get_battery_psc(splines=splines, E_b=E_b)
    pi_estimate = solve_for_pi(p_sc=p_sc_estimate, p_b=p_b)

    return pi_estimate

def get_next_Eb(p_b, E_b, splines, delta_t):
    """
    Get next step battery state of charge based on external power battery during delta_t

    :parram p_b: float with external battery power
    :param E_b: float with current battery state of charge
    :param splines: np.array with shape (n_splines, 2)
    :param delta_t: float with delta_t value

    :return next_Eb: next battery state of charge
    """
    p_i_estimate = get_battery_pi(p_b=p_b, E_b=E_b, splines=splines)
    
    return E_b - p_i_estimate * delta_t

def get_actual_pg(p_ev, p_s, p_b):
    """
    Calculate actual power grid based on actual data (ev, solar) and power battery

    :parram p_ev: float with actual ev power load
    :param p_s: float with actual solar power generation
    :param p_b: float with external batery power

    :return actual_pg: actual grid power
    """
    
    return p_ev - p_s - p_b

def get_actual_cost(p_g, price_buy, price_sell):
    """
    Calculate actual cost based on actual power grid

    :parram p_g: float with grid power
    :param price_buy: float with power price of buy
    :param price_sell: float with power price of sell

    :return actual_cost: actual power grid cost
    """
    return p_g * price_buy if p_g >= 0 else p_g * price_sell

def get_actual_emission(p_g, emission_factor):
    """
    Calculate emissions based on actual power grid

    :parram p_g: float with grid power
    :param emission_factor: float with emission factor [kg of CO2/kW]

    :return emission: actual emissions [kg of CO2]
    """
    return p_g * emission_factor

def season_datetimes(season, yearTest=2021, tz='America/Los_Angeles'):
    """
    Get all datetimes in the test set

    return datetimes_season: pd.DatetimeIndex with all datetimes in the test set
    """
    if season == 'winter':
        start_date = pd.Timestamp(year=yearTest, month=1, day=1, tz=tz)
        end_date = pd.Timestamp(year=yearTest, month=3, day=31, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    elif season == 'spring':
        start_date = pd.Timestamp(year=yearTest, month=4, day=1, tz=tz)
        end_date = pd.Timestamp(year=yearTest, month=6, day=30, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    elif season == 'summer':
        start_date = pd.Timestamp(year=yearTest, month=7, day=1, tz=tz)
        end_date = pd.Timestamp(year=yearTest, month=9, day=30, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    elif season == 'autumn':
        start_date = pd.Timestamp(year=yearTest, month=10, day=1, tz=tz)
        end_date = pd.Timestamp(year=yearTest, month=12, day=31, tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    
    datetimes_season = pd.date_range(start=start_date, end=end_date, freq='15min')

    if season == 'winter':
        return datetimes_season[datetimes_season.date != (pd.to_datetime(f'{yearTest}-03-14').date())]
    if season == 'summer':
        return datetimes_season[(datetimes_season.date != (pd.to_datetime(f'{yearTest}-07-16').date())) & (datetimes_season.date != (pd.to_datetime(f'{yearTest}-07-17').date()))]
    if season == 'autumn':
        return datetimes_season[(datetimes_season.date != (pd.to_datetime(f'{yearTest}-11-07').date())) & (datetimes_season.date != (pd.to_datetime(f'{yearTest}-10-27').date()))]
    else:
        return datetimes_season
    
def filter_eval_datetime(n_days_eval_per_season, n_days_rolling_feature, yearTest=2021, tz='America/Los_Angeles'):
    """
    Filter datetimes for evaluation

    :param n_days_eval_per_season: number of days to evaluate per season
    :param n_days_rolling_feature: number of days to use in the rolling feature

    :return eval_datetimes: pd.DatetimeIndex with datetimes to evaluate
    """
    eval_datetimes = pd.DatetimeIndex([])
    for s in ['winter', 'spring', 'summer', 'autumn']:
        s_dt = season_datetimes(season=s, yearTest=yearTest, tz=tz)
        eval_datetimes = eval_datetimes.append(s_dt[0:96*(n_days_rolling_feature + n_days_eval_per_season + 1)])

    return eval_datetimes
    