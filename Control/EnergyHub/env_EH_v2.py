
import numpy as np
import pandas as pd
import gymnasium as gym
import warnings
import pickle
import mapie
from EnergyHub.utils import * # to update internal state

class EHv2(gym.Env):
    """ 
    Energy Hub v2:
    """

    def __init__(self, battery_config, agent_config, data_dir='../Forecasting/EnergyHubEval/', model_dir='../not_uploaded/', env_mode='eval', do_save_env_vars=False):
        """
        Initialize the Energy Hub based on battery and agent specifications

        :param battery_config: dictionary with battery specifications
        :param agent_config: dictionary with agent specifications
        :param data_dir: folder to read forecast, actual and price data
        :param model_dir: folder to read trained models
        :param env_mode: 'train' or 'eval' to allow implementation of RL agents
        :param do_save_env_vars: True saves env variables for plotting
        """
        super().__init__()

        # eval settings
        self.priceYearTest = 2021 # 2021
        self.price_sell_prop_of_buy = 1 # 0.7
        self.scale_pv = 1
        self.scale_ev = 1

        # env settings
        self.env_mode = env_mode # mpc only uses env_mode='eval'
        self.delta_t = 0.25
        self.n_days_per_episode = agent_config['n_eval_days_per_episode'] # 1 for daily episodes and 7 for weekly episodes
        self.episode_length = np.int64(24 * self.n_days_per_episode / self.delta_t) # steps to take per episode
        self.episode_length_1H = np.int64(self.episode_length * self.delta_t) # steps to take per episode in 1H granularity
        self.time_k = None # time step within each episode
        self.n_episodes_per_season = agent_config['n_eval_episodes_per_season']
        self.n_episodes = self.n_episodes_per_season * 4
        self.episode_id = 0 # episode id of the evaluation
        self.season_id = 0 # season id of the evaluation
        self.season_episode_id = 0 # episode id within each season
        
        # reading settings
        self.data_dir = data_dir
        self.n_days_rolling_feature = 1

        self.load_features_horizon = self.pv_features_horizon = self.load_forecast_horizon = self.pv_forecast_horizon = None
        self.load_features_episode = self.pv_features_episode = self.load_actual_episode = self.pv_actual_episode = None
        self.price_features_horizon = self.price_ref_horizon= self.price_buy_horizon = self.price_sell_horizon = None
        self.price_features_episode = self.price_ref_episode = self.price_actual_episode = None
        self.emission_actual_episode = None

        # battery settings
        self.E_b = None
        self.E_b_0 = battery_config['E_b_0']
        self.E_b_min = battery_config['E_b_min']
        self.E_b_max = battery_config['E_b_max']
        self.splines = battery_config['splines']
        self.E_b_zero = agent_config['norm_E_b_zero']

        # prediction settings
        self.forecast_model_version = 'v2'
        self.forecast_mode = 'actual' if agent_config['version'] == 'MPComn' else 'pred'
        if self.forecast_mode == 'pred':
            self.forecast_mode = 'prob' if len(agent_config['alphas']) > 0 else 'det'
            self.ev_model = pickle.load(open(f'{model_dir}Pev_MapieGBR{self.forecast_model_version}_yearTest2021.pkl', 'rb'))
            self.pv_model = pickle.load(open(f'{model_dir}Ppv_MapieGBR{self.forecast_model_version}_yearTest2021.pkl', 'rb'))
            self.price_model = pickle.load(open(f'{model_dir}Cda_MapieGBR{self.forecast_model_version}_yearTest{self.priceYearTest}.pkl', 'rb'))

        self.alphas = agent_config['alphas']
        self.n_scenarios_per_var = np.int64(len(self.alphas) * 2)
        self.n_scenarios = np.int64(self.n_scenarios_per_var**3)

        # mpc settings
        self.horizon_optimization = np.int64(24 / self.delta_t)
        self.horizon_optimization_1H = np.int64(self.horizon_optimization / 4)
        self.prob_array = np.array([1 / self.n_scenarios] * self.n_scenarios)
        self.price_s_idx, self.pv_s_idx, self.ev_s_idx = self._generate_scenario_idx()

        # saving settings
        self.saving = do_save_env_vars
        self.grid_actual_save = self.load_actual_save = self.pv_actual_save = self.price_actual_save = self.reward_actual_save = self.emission_actual_save = None

        # initialize datasets
        self.all_data_ev = self.all_data_pv = self.all_data_price = self.all_data_emission = None
        self.all_data_ev, self.all_data_pv = self._read_power_data()
        self.all_data_price = self._read_price_data()
        self.all_data_emission = self._read_emission_data()

    def reset(self):
        """ 
        Reset environment

        :return observation: battery state of charge at time 0 (E_b[0])
        :return info: None
        """
        assert self.episode_id <= self.n_episodes - 1

        # reset episode
        self._init_episode_data()

        if self.saving:
            self.grid_actual_save = np.zeros(self.episode_length)
            self.reward_actual_save = np.zeros(self.episode_length)
            self.load_actual_save = self.load_actual_episode
            self.pv_actual_save = self.pv_actual_episode
            self.price_actual_save = self.price_actual_episode
            self.emission_actual_save = self.emission_actual_episode

        # forecast horizon
        self.time_k = 0
        self.load_features_horizon, self.pv_features_horizon, self.price_features_horizon, self.price_ref_horizon = self._filter_features(t_now=self.time_k)
        self.load_forecast_horizon, self.pv_forecast_horizon, self.price_buy_horizon = self._forecast(t_now=self.time_k)
        self.price_sell_horizon = self.price_buy_horizon * self.price_sell_prop_of_buy

        # set battery state of charge for time_k = 0
        observation = self.E_b_0 * self.E_b_zero
        self.E_b = np.clip(observation, self.E_b_min, self.E_b_max)

        # increment episode_id
        if self.n_episodes > 1:
            self.episode_id += 1 
        
        # increment episode id for multiseason evaluations 
        if self.season_episode_id < self.n_episodes_per_season - 1:
            self.season_episode_id += 1
        else:
            self.season_episode_id = 0
            self.season_id += 1

        info = None

        return observation, info

    def step(self, action):
        """ 
        Calculate the new system state and reward for the action and actual data (ev load, pv). Increment step within episode (time_k)

        :param action: agent action at time k (p_b[k])

        :return observation: updated battery state of charge at time k+1 (E_b[k+1])
        :return rewards: array with actual cost (cost_g[k]) and emission, based on the actual grid power at time k 
        :return done: flag indicating terminal step of episode
        :return truncated: False
        :return info: None
        """
        # estimate battery state of charge at time_k + 1
        observation = get_next_Eb(p_b=action, E_b=self.E_b, splines=self.splines, delta_t=self.delta_t)
        self.E_b = np.clip(observation, self.E_b_min, self.E_b_max) # clip to avoid infeasible optimal problem with the agent

        # get actual data and reward at time_k
        load_actual, pv_actual, price_buy_now, price_sell_now, emission_now = self._filter_actual_values(t_now=self.time_k)
        grid_actual = get_actual_pg(p_ev=load_actual, p_s=pv_actual, p_b=action)
        reward = get_actual_cost(p_g=grid_actual, price_buy=price_buy_now, price_sell=price_sell_now)
        emission = get_actual_emission(p_g=grid_actual, emission_factor=emission_now)

        if self.saving:
            self.grid_actual_save[self.time_k] = grid_actual
            self.reward_actual_save[self.time_k] = reward
            # self.emission_actual_save[self.time_k] = emission
        
        # terminate episode or slide horizon forward
        done = False if self.time_k < self.episode_length - 1 else True
        if not done:
            self.time_k += 1
            self.load_features_horizon, self.pv_features_horizon, self.price_features_horizon, self.price_ref_horizon = self._filter_features(t_now=self.time_k)
            self.load_forecast_horizon, self.pv_forecast_horizon, self.price_buy_horizon = self._forecast(t_now=self.time_k)
            self.price_sell_horizon = self.price_buy_horizon * self.price_sell_prop_of_buy

        truncated = False
        info = None

        return observation, np.array([reward, emission]), done, truncated, info
    
    
    def _read_power_data(self):
        """
        Read power data and features

        :return load: pd.DataFrame with load data and features
        :return pv: pd.DataFrame with pv data and features
        """
        ev = pd.read_csv(self.data_dir + 'AllTest_ev_2021_v0.csv')
        pv = pd.read_csv(self.data_dir + 'AllTest_pv_2021_v0.csv')

        ev['datetime'] = pd.to_datetime(ev['datetime'], utc=True).dt.tz_convert('America/Los_Angeles')
        pv['datetime'] = pd.to_datetime(ev['datetime'], utc=True).dt.tz_convert('America/Los_Angeles')

        read_eval_days_per_season = self.n_episodes_per_season * self.n_days_per_episode

        eval_dts = filter_eval_datetime(n_days_eval_per_season=read_eval_days_per_season, n_days_rolling_feature=self.n_days_rolling_feature)

        ev = ev[ev['datetime'].isin(eval_dts)]
        pv = pv[pv['datetime'].isin(eval_dts)]

        ev.reset_index(drop=True, inplace=True)
        pv.reset_index(drop=True, inplace=True)

        return ev, pv
    
    def _read_price_data(self):
        """
        Read price data and features

        :return price: pd.DataFrame with price data and features
        """
        price = pd.read_csv(self.data_dir + f'AllTest_Cda_{self.priceYearTest}_v0.csv')
        
        price['datetime'] = pd.to_datetime(price['datetime'], utc=True).dt.tz_convert('Europe/Amsterdam')
        read_eval_days_per_season = self.n_episodes_per_season * self.n_days_per_episode

        eval_dts = filter_eval_datetime(n_days_eval_per_season=read_eval_days_per_season, n_days_rolling_feature=self.n_days_rolling_feature, yearTest=self.priceYearTest, tz='Europe/Amsterdam')
        price = price[price['datetime'].isin(eval_dts)]
        price['price'] = np.clip(price['price'], a_min=5, a_max=None) / 1000 * self.delta_t  # euro/kW

        price.reset_index(drop=True, inplace=True)

        return price
    
    def _read_emission_data(self):
        """
        Read emission data

        :return emission: pd.DataFrame with emission data
        """
        emission = pd.read_csv(self.data_dir + f'AllTest_ghg_{self.priceYearTest}_v0.csv')
        
        emission['datetime'] = pd.to_datetime(emission['datetime'], utc=True).dt.tz_convert('Europe/Amsterdam')
        read_eval_days_per_season = self.n_episodes_per_season * self.n_days_per_episode

        eval_dts = filter_eval_datetime(n_days_eval_per_season=read_eval_days_per_season, n_days_rolling_feature=self.n_days_rolling_feature, yearTest=self.priceYearTest, tz='Europe/Amsterdam')
        emission = emission[emission['datetime'].isin(eval_dts)]
        emission['emissionfactor'] = emission['emissionfactor_kgkWh'] * self.delta_t  # kgCO2/kW

        emission.reset_index(drop=True, inplace=True)

        return emission

    def _init_episode_data(self):
        """
        Initialize episode by slicing actual data and features
        """
        # actual data
        season_idx = self.season_id * (self.n_episodes_per_season * self.n_days_per_episode + self.n_days_rolling_feature + 1) * 96
        season_idx_1H = np.int64(season_idx / 4)
        rollingF_idx = self.n_days_rolling_feature * 96
        rollingF_idx_1H = np.int64(rollingF_idx / 4)
        season_episode_idx = (self.season_episode_id * self.n_days_per_episode) * 96
        season_episode_idx_1H = np.int64(season_episode_idx / 4)

        self.load_actual_episode = self.all_data_ev.iloc[(season_idx+rollingF_idx+season_episode_idx):(season_idx+rollingF_idx+season_episode_idx+self.episode_length+self.horizon_optimization), 
                                                         self.all_data_ev.columns.get_loc('Power')].copy()
        self.pv_actual_episode = self.all_data_pv.iloc[(season_idx+rollingF_idx+season_episode_idx):(season_idx+rollingF_idx+season_episode_idx+self.episode_length+self.horizon_optimization), 
                                                        self.all_data_pv.columns.get_loc('powerkW')].copy()
        self.price_actual_episode = self.all_data_price.iloc[(season_idx_1H+rollingF_idx_1H+season_episode_idx_1H):(season_idx_1H+rollingF_idx_1H+season_episode_idx_1H+self.episode_length_1H+self.horizon_optimization_1H),
                                                             self.all_data_price.columns.get_loc('price')].copy()
        self.emission_actual_episode = self.all_data_emission.iloc[(season_idx_1H+rollingF_idx_1H+season_episode_idx_1H):(season_idx_1H+rollingF_idx_1H+season_episode_idx_1H+self.episode_length_1H+self.horizon_optimization_1H),
                                                             self.all_data_emission.columns.get_loc('emissionfactor')].copy()
        
        self.load_actual_episode.reset_index(drop=True, inplace=True)
        self.pv_actual_episode.reset_index(drop=True, inplace=True)
        self.price_actual_episode.reset_index(drop=True, inplace=True)
        self.emission_actual_episode.reset_index(drop=True, inplace=True)

        self.load_actual_episode = self.load_actual_episode * self.scale_ev
        self.pv_actual_episode = self.pv_actual_episode * self.scale_pv

        # features
        self.load_features_episode = self.all_data_ev.iloc[(season_idx+season_episode_idx):(season_idx+rollingF_idx+season_episode_idx+self.episode_length+self.horizon_optimization), 
                                                            :].copy() # ev_pred() needs initial rolling features
        self.pv_features_episode = self.all_data_pv.iloc[(season_idx+rollingF_idx+season_episode_idx):(season_idx+rollingF_idx+season_episode_idx+self.episode_length+self.horizon_optimization), 
                                                            :].copy() # pv_pred() does not need initial rolling features
        self.price_features_episode = self.all_data_price.iloc[(season_idx_1H+rollingF_idx_1H+season_episode_idx_1H):(season_idx_1H+rollingF_idx_1H+season_episode_idx_1H+self.episode_length_1H+self.horizon_optimization_1H), 
                                                            :].copy() # da_pred() does not need initial rolling features

        self.load_features_episode.drop(columns=['datetime'], inplace=True) # ev_pred() needs the 'Power' from the past to update the rolling features
        self.pv_features_episode.drop(columns=['datetime', 'powerkW'], inplace=True)
        self.price_ref_episode = self.price_features_episode.loc[:, ['priceRef']]
        self.price_features_episode.drop(columns=['datetime', 'price', 'priceRef', 'priceDif1'], inplace=True)

        self.load_features_episode.reset_index(drop=True, inplace=True)
        self.pv_features_episode.reset_index(drop=True, inplace=True)
        self.price_features_episode.reset_index(drop=True, inplace=True)

    def _filter_actual_values(self, t_now):
        """
        Filter actual values

        :param t_now: int with current time_k of the episode

        :return load_actual: filtered load actual data
        :return pv_actual: filtered pv actual data
        :return price_buy_actual: filtered p_buy actual data
        :return price_sell_actual: filtered p_sell actual data
        :return emission_now: filtered emission actual data
        """
        load_actual = self.load_actual_episode.iloc[t_now]
        pv_actual = self.pv_actual_episode.iloc[t_now]
        price_buy_actual = self.price_actual_episode.iloc[np.int64(t_now/4)]
        price_sell_actual = price_buy_actual * self.price_sell_prop_of_buy
        emission_now = self.emission_actual_episode.iloc[np.int64(t_now/4)]

        return load_actual, pv_actual,  price_buy_actual, price_sell_actual, emission_now
    
    def _filter_features(self, t_now):
        """
        Filter features

        :param t_now: int with current time_k of the episode

        :return load_features_horizon: filtered load features
        :return pv_features_horizon: filtered pv features
        :return price_features_horizon: filtered price features
        :return price_ref_horizon: filtered price ref for differences
        """
        rollingF_idx = self.n_days_rolling_feature * 96

        if self.forecast_mode != 'actual':
            load_features_horizon = self.load_features_episode.iloc[t_now:t_now+rollingF_idx+self.horizon_optimization, :].copy()
            pv_features_horizon = self.pv_features_episode.iloc[t_now:t_now+self.horizon_optimization, :].copy()
            price_features_horizon = self.price_features_episode.iloc[np.int64(t_now/4):np.int64(t_now/4)+self.horizon_optimization_1H+1, :].copy()
            price_ref_horizon = self.price_ref_episode.iloc[np.int64(t_now/4):np.int64(t_now/4)+self.horizon_optimization_1H+1, :].copy()

        else:
            load_features_horizon = None
            pv_features_horizon = None
            price_features_horizon = None
            price_ref_horizon = None
        
        return load_features_horizon, pv_features_horizon, price_features_horizon, price_ref_horizon

    def _forecast(self, t_now):
        """
        Create forecasts for the optimization horizon

        :param t_now: int with current time_k of the episode

        :return load_forecast_horizon: np.array with load forecast
        :return pv_forecast_horizon: np.array with pv forecast
        :return price_forecast_horizon: np.array with price forecast
        """
        if self.forecast_mode != 'actual':
            if self.forecast_mode == 'det':
                load_forecast_horizon = self._ev_pred_det(model=self.ev_model, X=self.load_features_horizon, n_ahead=self.horizon_optimization) * self.scale_ev
                pv_forecast_horizon = self._pv_pred_det(model=self.pv_model, X=self.pv_features_horizon, n_ahead=self.horizon_optimization) * self.scale_pv
                price_forecast_horizon = self._da_pred_det_15min(model=self.price_model, X=self.price_features_horizon, y_ref=self.price_ref_horizon, n_ahead=self.horizon_optimization_1H+1)
            else:
                load_forecast_horizon = self._ev_pred_prob(model=self.ev_model, alphas=self.alphas, X=self.load_features_horizon, n_ahead=self.horizon_optimization, only_alphas=False) * self.scale_ev
                pv_forecast_horizon = self._pv_pred_prob(model=self.pv_model, alphas=self.alphas, X=self.pv_features_horizon, n_ahead=self.horizon_optimization,  only_alphas=False) * self.scale_pv
                price_forecast_horizon = self._da_pred_prob_15min(model=self.price_model, alphas=self.alphas, X=self.price_features_horizon, y_ref=self.price_ref_horizon, n_ahead=self.horizon_optimization_1H+1,  only_alphas=False)
        else:
            load_forecast_horizon = np.array(self.load_actual_episode.iloc[t_now:t_now+self.horizon_optimization])
            pv_forecast_horizon = np.array(self.pv_actual_episode.iloc[t_now:t_now+self.horizon_optimization])
            price_forecast_horizon = np.repeat(np.array(self.price_actual_episode.iloc[np.int64(t_now/4):np.int64(t_now/4)+self.horizon_optimization_1H+1]), 4)
         
        price_forecast_horizon = price_forecast_horizon[(t_now%4):self.horizon_optimization+(t_now%4)]
        
        return load_forecast_horizon, pv_forecast_horizon, price_forecast_horizon
    
    def _da_pred_det_15min(self, model, X, y_ref, n_ahead=24):
        """
        Predict DA price with 15min granularity (Deterministic)

        :param model: trained model to predict DA price with 1h granularity
        :param X: pd.DataFrame with features for the horizon with 1h granularity
        :param y_ref: pd.Series with reference DA price with 1h granularity
        :param n_ahead: number of steps in the horizon with 1h granularity

        :return y_preds: np.array of predicted DA price with 15min granularity
        """
        y_preds = np.zeros(n_ahead)
        X.reset_index(drop=True, inplace=True)

        X.loc[:, 'isoDay_std'] = X['isoDay_std'].iloc[0] # only the first value of isoDay_sum is known

        warnings.filterwarnings("ignore")
        pred, _ = model.predict(X, alpha=0.1, ensemble=False, allow_infinite_bounds=False, optimize_beta=False)
        y_preds = np.clip(pred[:] + y_ref['priceRef'], a_min=5, a_max=None)

        # return np.repeat(y_preds, 4)
        return np.repeat(np.array(y_preds), 4) / 1000 * self.delta_t # euro/kW

    def _pv_pred_det(self, model, X, n_ahead=96):
        """
        Predict PV power output for the horizon (Deterministic)

        :param model: trained model to predict PV power output
        :param X: pd.DataFrame with features for the horizon
        :param n_ahead: number of steps in the horizon

        :return y_preds: np.array with predicted PV power output
        """
        y_preds = np.zeros(n_ahead)
        X.reset_index(drop=True, inplace=True)

        X.loc[:, 'isoDay_sum'] = X['isoDay_sum'].iloc[0] # only the first value of isoDay_sum is known
        X.loc[:, 'isoDay_std'] = X['isoDay_std'].iloc[0]
        
        warnings.filterwarnings("ignore")
        pred, _ = model.predict(X, alpha=0.1, ensemble=False, allow_infinite_bounds=False, optimize_beta=False)
        y_preds[:] = np.clip(pred[:], a_min=0, a_max=None)

        y_preds[X[X['isDaylight']==False].index] = 0
        
        return y_preds
    
    def _ev_pred_det(self, model, X, n_ahead=96, n_daily_lags=96, min_step_lag=24):
        """
        Predict Total EV power demand over the horizon (Deterministic)

        :param model: trained model to predict EV power demand
        :param X: pd.DataFrame with features for the horizon
        :param n_ahead: number of steps in the horizon
        :param n_daily_lags: number of previous steps used to calculate rolling features
        :param min_step_lag: number of steps that can be predicted without update on rolling features

        :return y_preds: np.array of predicted total EV power demand
        """
        y_preds = np.zeros(n_ahead)
        X.reset_index(drop=True, inplace=True)

        id_start = n_daily_lags
        X_rolling_features_segment = X.loc[0:id_start, ['Power']].copy()
        X_rolling_features_segment.reset_index(drop=True, inplace=True)

        X_test_segment = X.iloc[id_start:id_start+n_ahead].copy()
        X_test_segment.loc[:, 'isoDay_sum'] = X_test_segment['isoDay_sum'].iloc[0] # only the first value of isoDay_sum is known
        X_test_segment.loc[:, 'isoDay_std'] = X_test_segment['isoDay_std'].iloc[0] 
        X_test_segment.drop(columns=['Power'], inplace=True) # remove Power from X_test
        
        for step in range(0, n_ahead, min_step_lag):

            warnings.filterwarnings("ignore")
            pred, _ = model.predict(X_test_segment.iloc[step:step+min_step_lag], alpha=0.1, ensemble=False, allow_infinite_bounds=False, optimize_beta=False)
            y_preds[step:step+min_step_lag] = np.clip(pred[:], a_min=0, a_max=None)

            # recompute rolling features
            if step + min_step_lag < n_ahead: 
                X_rolling_features_segment = pd.concat([X_rolling_features_segment.iloc[min_step_lag:], X_rolling_features_segment.iloc[-min_step_lag:]], ignore_index=True)
                X_rolling_features_segment.iloc[-min_step_lag:, X_rolling_features_segment.columns.get_loc('Power')] = pred[:]

                if step + min_step_lag < len(X_test_segment):
                    X_rolling_features_test_segment = pd.concat([X_rolling_features_segment, X_test_segment.iloc[step+min_step_lag:step+2*min_step_lag, :]], ignore_index=True)
                    X_rolling_features_test_segment.fillna(0, inplace=True)
                    X_rolling_features_test_segment['H-6'] = X_rolling_features_test_segment['Power'].rolling(window=7*4).sum() - X_rolling_features_test_segment['Power'].rolling(window=6*4).sum()
                    X_rolling_features_test_segment['H-23'] = X_rolling_features_test_segment['Power'].rolling(window=24*4).sum() - X_rolling_features_test_segment['Power'].rolling(window=23*4).sum()
                    X_test_segment.iloc[step+min_step_lag:step+2*min_step_lag, X_test_segment.columns.get_loc('H-6')] = X_rolling_features_test_segment.iloc[-min_step_lag:, X_rolling_features_test_segment.columns.get_loc('H-6')]
                    X_test_segment.iloc[step+min_step_lag:step+2*min_step_lag, X_test_segment.columns.get_loc('H-23')] = X_rolling_features_test_segment.iloc[-min_step_lag:, X_rolling_features_test_segment.columns.get_loc('H-23')]

        return y_preds

    def _generate_scenario_idx(self):
        """
        Generate all combinations of scenarios indexes per variable (stochastic tree with independence assumption)

        :return scenario_idx_price: np.arrary with scenario indexes variables price
        :return scenario_idx_pv: np.arrary with scenario indexes variables pv
        :return scenario_idx_ev: np.arrary with scenario indexes variables ev
        """
        scenario_idx_price = np.zeros(self.n_scenarios, dtype=int)
        scenario_idx_pv = np.zeros(self.n_scenarios, dtype=int)
        scenario_idx_ev = np.zeros(self.n_scenarios, dtype=int)

        for idx_price in range(self.n_scenarios_per_var):
            for idx_pv in range(self.n_scenarios_per_var):
                for idx_ev in range(self.n_scenarios_per_var):
                    scenario_idx = idx_price * self.n_scenarios_per_var**2 + idx_pv * self.n_scenarios_per_var + idx_ev
                    scenario_idx_price[scenario_idx] = idx_price
                    scenario_idx_pv[scenario_idx] = idx_pv
                    scenario_idx_ev[scenario_idx] = idx_ev

        return scenario_idx_price, scenario_idx_pv, scenario_idx_ev

    def _da_pred_prob_15min(self, model, alphas, X, y_ref, n_ahead=24, only_alphas=True):
        """
        Predict DA price with 15min granularity (Probabilistic)

        :param model: trained model to predict DA price with 1h granularity
        :param alphas: list of alpha values for the prediction intervals    
        :param X: pd.DataFrame with features for the horizon with 1h granularity
        :param y_ref: pd.Series with reference DA price with 1h granularity
        :param n_ahead: number of steps in the horizon with 1h granularity
        :param only_alphas: boolean to return only the alphas or all scenarios

        :return y_all: np.array of predicted DA price with 15min granularity
        """
        y_preds = np.zeros(n_ahead)
        y_cis = np.zeros((n_ahead, len(alphas) * 2))
        y_all = np.zeros((n_ahead, len(alphas) * 2 + 1))
        X.reset_index(drop=True, inplace=True)

        X.loc[:, 'isoDay_std'] = X['isoDay_std'].iloc[0] # only the first value of isoDay_f is known

        warnings.filterwarnings("ignore")
        for a in range(len(alphas)):
            pred, ci = model.predict(X, alpha=alphas[a], ensemble=True, allow_infinite_bounds=False, optimize_beta=False)
            y_preds[:] = np.clip(pred[:] + y_ref['priceRef'], a_min=5, a_max=None)
            y_cis[:, a*2:(a+1)*2] = np.clip(ci[:, :, 0]  + np.array([y_ref['priceRef'].values, y_ref['priceRef'].values]).T , a_min=5, a_max=None)

        y_all[:, :] = np.hstack([y_preds.reshape(-1, 1), y_cis[:, :]])

        if only_alphas: 
            return np.repeat(y_all[:, 1:], 4, axis=0) / 1000 * self.delta_t # euro/kW
        else:
            return np.repeat(y_all, 4, axis=0) / 1000 * self.delta_t # euro/kW
        
    def _pv_pred_prob(self, model, alphas, X, n_ahead=96, only_alphas=True):
        """
        Predict PV power output for the horizon (Probabilistic)

        :param model: trained model to predict PV power output
        :param alphas: list of alpha values for the prediction intervals    
        :param X: pd.DataFrame with features for the horizon
        :param n_ahead: number of steps in the horizon
        :param only_alphas: boolean to return only the alphas or all scenarios

        :return y_all: np.array with predicted PV power output
        """
        y_preds = np.zeros(n_ahead)
        y_cis = np.zeros((n_ahead, len(alphas) * 2))
        y_all = np.zeros((n_ahead, len(alphas) * 2 + 1))
        X.reset_index(drop=True, inplace=True)

        X.loc[:, 'isoDay_sum'] = X['isoDay_sum'].iloc[0] # only the first value of isoDay_f is known
        X.loc[:, 'isoDay_std'] = X['isoDay_std'].iloc[0]
        
        warnings.filterwarnings("ignore")
        for a in range(len(alphas)):
            pred, ci = model.predict(X, alpha=alphas[a], ensemble=True, allow_infinite_bounds=False, optimize_beta=False)
            y_preds[:] = np.clip(pred[:], a_min=0, a_max=None)
            y_cis[:, a*2:(a+1)*2] = np.clip(ci[:, :, 0], a_min=0, a_max=None)

        y_all[:, :] = np.hstack([pred.reshape(-1, 1), y_cis[:, :]]) 
        y_all[X[X['isDaylight']==False].index, :] = 0

        if only_alphas:
            return y_all[:, 1:]
        else:
            return y_all
        
    def _ev_pred_prob(self, model, alphas, X, n_ahead=96, n_daily_lags=96, min_step_lag=24, only_alphas=True):
        """
        Predict Total EV power demand over the horizon (Probabilistic)

        :param model: trained model to predict EV power demand
        :param alphas: list of alpha values for the prediction intervals    
        :param X: pd.DataFrame with features for the horizon
        :param n_ahead: number of steps in the horizon
        :param n_daily_lags: number of previous steps used to calculate rolling features
        :param min_step_lag: number of steps that can be predicted without update on rolling features
        :param only_alphas: boolean to return only the alphas or all scenarios

        :return y_all: np.array of predicted total EV power demand
        """        
        y_preds = np.zeros(n_ahead)
        y_cis = np.zeros((n_ahead, len(alphas) * 2))
        y_all = np.zeros((n_ahead, len(alphas) * 2 + 1))
        X.reset_index(drop=True, inplace=True)

        id_start = n_daily_lags
        X_rolling_features_segment = X.loc[0:id_start, ['Power']].copy()
        X_rolling_features_segment.reset_index(drop=True, inplace=True)

        X_test_segment = X.iloc[id_start:id_start+n_ahead].copy()
        X_test_segment.loc[:, 'isoDay_sum'] = X_test_segment['isoDay_sum'].iloc[0] # only the first value of isoDay_sum is known
        X_test_segment.loc[:, 'isoDay_std'] = X_test_segment['isoDay_std'].iloc[0] 
        X_test_segment.drop(columns=['Power'], inplace=True) # remove Power from X_test
        
        for step in range(0, n_ahead, min_step_lag):

            warnings.filterwarnings("ignore")
            for a in range(len(alphas)):
                pred, ci = model.predict(X_test_segment.iloc[step:step+min_step_lag], alpha=alphas[a], ensemble=True, allow_infinite_bounds=False, optimize_beta=False)
                y_preds[step:step+min_step_lag] = np.clip(pred[:], a_min=0, a_max=None)
                y_cis[step:step+min_step_lag, a*2:(a+1)*2] = np.clip(ci[:, :, 0], a_min=0, a_max=None)

            y_all[step:step+min_step_lag, :] = np.hstack([pred.reshape(-1, 1), y_cis[step:step+min_step_lag, :]]) 

            # recompute rolling features
            if step + min_step_lag < n_ahead: 
                X_rolling_features_segment = pd.concat([X_rolling_features_segment.iloc[min_step_lag:], X_rolling_features_segment.iloc[-min_step_lag:]], ignore_index=True)
                X_rolling_features_segment.iloc[-min_step_lag:, X_rolling_features_segment.columns.get_loc('Power')] = pred[:]

                if step + min_step_lag < len(X_test_segment):
                    X_rolling_features_test_segment = pd.concat([X_rolling_features_segment, X_test_segment.iloc[step+min_step_lag:step+2*min_step_lag, :]], ignore_index=True)
                    X_rolling_features_test_segment.fillna(0, inplace=True)
                    X_rolling_features_test_segment['H-6'] = X_rolling_features_test_segment['Power'].rolling(window=7*4).sum() - X_rolling_features_test_segment['Power'].rolling(window=6*4).sum()
                    X_rolling_features_test_segment['H-23'] = X_rolling_features_test_segment['Power'].rolling(window=24*4).sum() - X_rolling_features_test_segment['Power'].rolling(window=23*4).sum()
                    X_test_segment.iloc[step+min_step_lag:step+2*min_step_lag, X_test_segment.columns.get_loc('H-6')] = X_rolling_features_test_segment.iloc[-min_step_lag:, X_rolling_features_test_segment.columns.get_loc('H-6')]
                    X_test_segment.iloc[step+min_step_lag:step+2*min_step_lag, X_test_segment.columns.get_loc('H-23')] = X_rolling_features_test_segment.iloc[-min_step_lag:, X_rolling_features_test_segment.columns.get_loc('H-23')]

        if only_alphas:
            return y_all[:, 1:]
        else:
            return y_all