
import cvxpy as cp

class MPCdetv0week:
    """
    MPCdetv0: SOCP formualtion
    """

    def __init__(self, env, agent_config, do_save_agent_vars=True):
        """ 
        Initialize MPC deterministic agent 

        :param env: gym.Env object. Energy Hub environment
        :param agent_config: dictionary with agent specifications
        :param do_save_agent_vars: True saves agent variables for plotting
        """

        # mpc settings
        self.opt_problem = None

        # normalization parameters
        self.p_zero = agent_config['norm_p_zero']
        self.p_sc_zero = agent_config['norm_p_sc_zero']
        self.E_b_zero = agent_config['norm_E_b_zero']
        self.cost_zero = agent_config['norm_cost_zero']

        # battery variables
        self.p_b_n = cp.Variable((env.horizon_optimization))
        self.p_i_n = cp.Variable((env.horizon_optimization))
        self.p_sc_n = cp.Variable((env.horizon_optimization))
        self.E_b_n = cp.Variable((env.horizon_optimization + 1), nonneg=True)

        # grid variables
        self.p_g_n = cp.Variable((env.horizon_optimization))
        self.cost_g_n = cp.Variable((env.horizon_optimization))

        # agent variables to save
        self.saving = do_save_agent_vars
        self.p_b = {}
        self.p_i = {}
        self.p_g = {}
        self.E_b = {}
        self.res_load = {}
        self.ev = {}
        self.pv = {}
        self.price_buy = {}
        self.price_sell = {}

    def act(self, env, verbose):
        """ 
        Generate actions from MPC agent. Formulate and solve SOCP. Save agent variables.

        :param env: gym.Env object. Energy Hub environment
        :param verbose: boolean. Print info from cvxpy.Problem.solve() 

        :return action: return battery power from the first step of the horizon of optimization (p_b[0])
        """

        # formulate SOCP
        self._formulate(env=env)

        # solve SOCP
        result = self.opt_problem.solve(solver=cp.GUROBI, verbose=verbose)
        action = self.p_b_n.value[0] * self.p_zero

        # save agent variables
        if self.saving:
            self.p_b[env.time_k] = self.p_b_n.value * self.p_zero
            self.p_i[env.time_k] = self.p_i_n.value * self.p_zero
            self.p_g[env.time_k] = self.p_g_n.value * self.p_zero
            self.E_b[env.time_k] = self.E_b_n.value * self.E_b_zero
            self.res_load[env.time_k] = env.load_forecast_horizon - env.pv_forecast_horizon
            self.ev[env.time_k] = env.load_forecast_horizon
            self.pv[env.time_k] = env.pv_forecast_horizon
            self.price_buy[env.time_k] = env.price_buy_horizon
            self.price_sell[env.time_k] = env.price_sell_horizon

        return action

    def _formulate(self, env):
        """ 
        Formulate SOCP 

        :param env: gym.Env object. Energy Hub environment
        """

        constraints = []

        # battery constraints
        constraints += [ cp.SOC((self.p_i_n[:] - self.p_b_n[:] + self.p_sc_n[:] * self.p_sc_zero),
                                    cp.vstack([self.p_i_n[:] - self.p_b_n[:] - self.p_sc_n[:] * self.p_sc_zero,
                                               (2 * self.p_i_n[:] * self.p_zero**0.5)]),
                                    axis=0) ]
        for spline in env.splines:
            constraints += [self.p_sc_n <= (spline[0] * self.E_b_n[:env.horizon_optimization] * self.E_b_zero + spline[1]) / self.p_sc_zero]

        constraints += [ self.E_b_n[0] == env.E_b / self.E_b_zero] # initial conditions
        if (env.time_k >= 0 + env.horizon_optimization * (env.n_days_per_episode - 1)):
            constraints += [ env.E_b_0 == self.E_b_n[env.horizon_optimization - (env.time_k - env.horizon_optimization * (env.n_days_per_episode - 1))] ] # periodicity. equality at the beginning and end of the episode (day, week, ...)
        constraints += [ self.E_b_n[:] >= env.E_b_min / self.E_b_zero ] # min capacity
        constraints += [ self.E_b_n[:] <= env.E_b_max / self.E_b_zero ] # max capacity

        constraints += [self.E_b_n[1:] == self.E_b_n[:-1] - self.p_i_n * (env.delta_t * self.p_zero / self.E_b_zero)]

        # load constraints
        constraints += [env.load_forecast_horizon[:] / self.p_zero ==
                        env.pv_forecast_horizon[:] / self.p_zero + self.p_b_n + self.p_g_n]

        # grid constraints
        price_buy = cp.Constant(env.price_buy_horizon[:])
        price_sell = cp.Constant(env.price_sell_horizon[:])
        constraints += [
            self.cost_g_n >= cp.multiply(self.p_g_n, self.p_zero * price_buy / self.cost_zero),
            self.cost_g_n >= cp.multiply(self.p_g_n, self.p_zero * price_sell / self.cost_zero)
        ]

        # electricity cost
        cost = cp.sum(self.cost_g_n) # grid buy and grid sell

        # problem definition
        self.opt_problem = cp.Problem(objective=cp.Minimize(cost), constraints=constraints)
    