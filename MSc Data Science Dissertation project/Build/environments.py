import datetime

from history import History
from portfolio import TargetPortfolio

import gymnasium as gym
from gymnasium import spaces

import pandas as pd

import numpy as np

import os



def basic_reward_function(history : History, target_profit):
    
    #Step reward
    #balance_change = history["portfolio_valuation", -1] - history["portfolio_valuation", -2]
    
    #immediate_change = balance_change/history["portfolio_valuation", 0]
    
    #if immediate_change >= target_profit:
    #    step_reward = 2
        
    #elif target_profit > immediate_change > 0: 
    #    step_reward = 1
    
    #elif 0 > immediate_change >= -target_profit/5: 
    #    step_reward = -1
    
    #elif  - target_profit/5 > immediate_change: 
    #    step_reward = -5
    
    #else: 
    #    step_reward = 0
        
    #Progress reward    
    #positive_reward_threshold = 0.03
    #negative_reward_threshold = -0.03
    
    #overall_balance_change = (history["portfolio_valuation", -1] - history["portfolio_valuation", 0]) / history["portfolio_valuation", 0]
    
    #if overall_balance_change > positive_reward_threshold:
    #    reward = 4.0
    #elif overall_balance_change < negative_reward_threshold:
    #    reward = -4.0
    #else:
    #    reward = 0.0
    
    balance_change = history["portfolio_valuation", -1] - history["portfolio_valuation", -2]
    
    immediate_change = balance_change/history["portfolio_valuation", 0]
    
    overall_change = (history["portfolio_valuation", -1] - history["portfolio_valuation", 0])/history["portfolio_valuation", 0]
    
    
    
    return immediate_change + overall_change
    

def dynamic_feature_last_position_taken(history):
    return history['position', -1]
    
def dynamic_feature_real_position(history):
    return history['real_position', -1]


#Env option 1
class TradingEnv(gym.Env):
    """
    An easy trading environment for OpenAI gym. It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('TradingEnv', ...)


    :param df: The market DataFrame. It must contain 'open', 'high', 'low', 'close'. Index must be DatetimeIndex. Your desired inputs need to contain 'feature' in their column name : this way, they will be returned as observation at each step.
    :type df: pandas.DataFrame

    :param positions: List of the positions allowed by the environment.
    :type positions: optional - list[int or float]

    :param dynamic_feature_functions: The list of the dynamic features functions. By default, two dynamic features are added :
    
        * the last position taken by the agent.
        * the real position of the portfolio (that varies according to the price fluctuations)

    :type dynamic_feature_functions: optional - list   

    :param reward_function: Take the History object of the environment and must return a float.
    :type reward_function: optional - function<History->float>

    :param windows: Default is None. If it is set to an int: N, every step observation will return the past N observations. It is recommended for Recurrent Neural Network based Agents.
    :type windows: optional - None or int

    :param trading_fees: Transaction trading fees (buy and sell operations). eg: 0.01 corresponds to 1% fees
    :type trading_fees: optional - float

    :param borrow_interest_rate: Borrow interest rate per step (only when position < 0 or position > 1). eg: 0.01 corresponds to 1% borrow interest rate per STEP ; if your know that your borrow interest rate is 0.05% per day and that your timestep is 1 hour, you need to divide it by 24 -> 0.05/100/24.
    :type borrow_interest_rate: optional - float

    :param portfolio_initial_value: Initial valuation of the portfolio.
    :type portfolio_initial_value: float or int

    :param initial_position: You can specify the initial position of the environment or set it to 'random'. It must contained in the list parameter 'positions'.
    :type initial_position: optional - float or int

    :param max_episode_duration: If a integer value is used, each episode will be truncated after reaching the desired max duration in steps (by returning `truncated` as `True`). When using a max duration, each episode will start at a random starting point.
    :type max_episode_duration: optional - int or 'max'

    :param verbose: If 0, no log is outputted. If 1, the env send episode result logs.
    :type verbose: optional - int
    
    :param name: The name of the environment (eg. 'BTC/USDT')
    :type name: optional - str
    
    """
    metadata = {'render_modes': ['logs']}
    def __init__(self,
                df : pd.DataFrame,
                positions : list = [0, 1],
                dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
                reward_function = basic_reward_function,
                windows = None,
                trading_fees = 0,
                borrow_interest_rate = 0,
                portfolio_initial_value = 1000,
                initial_position ='random',
                max_episode_duration = 'max',
                verbose = 1,
                name = "Stock",
                render_mode= "logs"
                ):
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        assert self.initial_position in self.positions or self.initial_position == 'random', "The 'initial_position' parameter must be 'random' or a position mentionned in the 'position' (default is [0, 1]) parameter."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode
        self._set_df(df)
        
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape = [self._nb_features]
        )
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.windows, self._nb_features]
            )
        
        self.log_metrics = []


    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])


    
    def _get_ticker(self, delta = 0):
        return self.df.iloc[self._idx + delta]
    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
    
    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        if self.windows is None:
            _step_index = self._idx
        else: 
            _step_index = np.arange(self._idx + 1 - self.windows , self._idx + 1)
        return self._obs_array[_step_index]

    
    def reset(self, seed = None, options=None):
        super().reset(seed = seed)
        
        self._step = 0
        self._position = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
        self._limit_orders = {}
        

        self._idx = 0
        if self.windows is not None: self._idx = self.windows - 1
        if self.max_episode_duration != 'max':
            self._idx = np.random.randint(
                low = self._idx, 
                high = len(self.df) - self.max_episode_duration - self._idx
            )
        
        self._portfolio  = TargetPortfolio(
            position = self._position,
            value = self.portfolio_initial_value,
            price = self._get_price()
        )
        
        self.historical_info = History(max_size= len(self.df))
        self.historical_info.set(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index =self.positions.index(self._position),
            position = self._position,
            real_position = self._position,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            portfolio_distribution = self._portfolio.get_portfolio_distribution(),
            reward = 0,
        )

        return self._get_obs(), self.historical_info[0]

    def render(self):
        pass

    def _trade(self, position, price = None):
        self._portfolio.trade_to_position(
            position, 
            price = self._get_price() if price is None else price, 
            trading_fees = self.trading_fees
        )
        self._position = position
        return

    def _take_action(self, position):
        if position != self._position:
            self._trade(position)
    
    def _take_action_order_limit(self):
        if len(self._limit_orders) > 0:
            ticker = self._get_ticker()
            for position, params in self._limit_orders.items():
                if position != self._position and params['limit'] <= ticker["high"] and params['limit'] >= ticker["low"]:
                    self._trade(position, price= params['limit'])
                    if not params['persistent']: del self._limit_orders[position]


    
    def add_limit_order(self, position, limit, persistent = False):
        self._limit_orders[position] = {
            'limit' : limit,
            'persistent': persistent
        }
    
    def step(self, position_index = None):
        if position_index is not None: 
            self._take_action(self.positions[position_index])
            
        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate= self.borrow_interest_rate)
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if isinstance(self.max_episode_duration,int) and self._step >= self.max_episode_duration - 1:
            truncated = True

        self.historical_info.add(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index =position_index,
            position = self._position,
            real_position = self._portfolio.real_position(price),
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = portfolio_value,
            portfolio_distribution = portfolio_distribution, 
            reward = 0
        )
        if not done:
            reward = self.reward_function(self.historical_info, self.initial_position*2)
            self.historical_info["reward", -1] = reward

        if done or truncated:
            self.calculate_metrics()
            self.log()
        return self._get_obs(),  self.historical_info["reward", -1], done, truncated, self.historical_info[-1]

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })
    def calculate_metrics(self):
        self.results_metrics = {
            "Market Return" : f"{100*(self.historical_info['data_close', -1] / self.historical_info['data_close', 0] -1):5.2f}%",
            "Portfolio Return" : f"{100*(self.historical_info['portfolio_valuation', -1] / self.historical_info['portfolio_valuation', 0] -1):5.2f}%",
        }

        for metric in self.log_metrics:
            self.results_metrics[metric['name']] = metric['function'](self.historical_info)
    def get_metrics(self):
        return self.results_metrics
    def log(self):
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

    def save_for_render(self, dir = "render_logs"):
        assert "open" in self.df and "high" in self.df and "low" in self.df and "close" in self.df, "Your DataFrame needs to contain columns : open, high, low, close to render !"
        columns = list(set(self.historical_info.columns) - set([f"date_{col}" for col in self._info_columns]))
        history_df = pd.DataFrame(
            self.historical_info[columns], columns= columns
        )
        history_df.set_index("date", inplace= True)
        history_df.sort_index(inplace = True)
        render_df = self.df.join(history_df, how = "inner")
        
        if not os.path.exists(dir):os.makedirs(dir)
        render_df.to_pickle(f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")


#Env option 2
class TradingEnvContinuous(gym.Env):
    """
    An easy trading environment for OpenAI gym. It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('TradingEnv', ...)


    :param df: The market DataFrame. It must contain 'open', 'high', 'low', 'close'. Index must be DatetimeIndex. Your desired inputs need to contain 'feature' in their column name : this way, they will be returned as observation at each step.
    :type df: pandas.DataFrame

    :param positions: List of the positions allowed by the environment.
    :type positions: optional - list[int or float]

    :param dynamic_feature_functions: The list of the dynamic features functions. By default, two dynamic features are added :
    
        * the last position taken by the agent.
        * the real position of the portfolio (that varies according to the price fluctuations)

    :type dynamic_feature_functions: optional - list   

    :param reward_function: Take the History object of the environment and must return a float.
    :type reward_function: optional - function<History->float>

    :param windows: Default is None. If it is set to an int: N, every step observation will return the past N observations. It is recommended for Recurrent Neural Network based Agents.
    :type windows: optional - None or int

    :param trading_fees: Transaction trading fees (buy and sell operations). eg: 0.01 corresponds to 1% fees
    :type trading_fees: optional - float

    :param borrow_interest_rate: Borrow interest rate per step (only when position < 0 or position > 1). eg: 0.01 corresponds to 1% borrow interest rate per STEP ; if your know that your borrow interest rate is 0.05% per day and that your timestep is 1 hour, you need to divide it by 24 -> 0.05/100/24.
    :type borrow_interest_rate: optional - float

    :param portfolio_initial_value: Initial valuation of the portfolio.
    :type portfolio_initial_value: float or int

    :param initial_position: You can specify the initial position of the environment or set it to 'random'. It must contained in the list parameter 'positions'.
    :type initial_position: optional - float or int

    :param max_episode_duration: If a integer value is used, each episode will be truncated after reaching the desired max duration in steps (by returning `truncated` as `True`). When using a max duration, each episode will start at a random starting point.
    :type max_episode_duration: optional - int or 'max'

    :param verbose: If 0, no log is outputted. If 1, the env send episode result logs.
    :type verbose: optional - int
    
    :param name: The name of the environment (eg. 'BTC/USDT')
    :type name: optional - str
    
    """
    metadata = {'render_modes': ['logs']}
    def __init__(self,
                df : pd.DataFrame,
                positions : list = [0, 1],
                dynamic_feature_functions = [dynamic_feature_last_position_taken, dynamic_feature_real_position],
                reward_function = basic_reward_function,
                windows = None,
                trading_fees = 0,
                borrow_interest_rate = 0,
                portfolio_initial_value = 1000,
                initial_position ='random',
                max_episode_duration = 'max',
                verbose = 1,
                name = "Stock",
                render_mode= "logs"
                ):
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        assert self.initial_position in self.positions or self.initial_position == 'random', "The 'initial_position' parameter must be 'random' or a position mentionned in the 'position' (default is [0, 1]) parameter."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode
        self._set_df(df)
        
        self.action_space = spaces.Box(low=self.positions[0], high=self.positions[-1], shape=[1], dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape = [self._nb_features]
        )
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.windows, self._nb_features]
            )
        
        self.log_metrics = []


    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])


    
    def _get_ticker(self, delta = 0):
        return self.df.iloc[self._idx + delta]
    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
    
    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        if self.windows is None:
            _step_index = self._idx
        else: 
            _step_index = np.arange(self._idx + 1 - self.windows , self._idx + 1)
        return self._obs_array[_step_index]

    
    def reset(self, seed = None, options=None):
        super().reset(seed = seed)
        
        self._step = 0
        self._position = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
        self._limit_orders = {}
        

        self._idx = 0
        if self.windows is not None: self._idx = self.windows - 1
        if self.max_episode_duration != 'max':
            self._idx = np.random.randint(
                low = self._idx, 
                high = len(self.df) - self.max_episode_duration - self._idx
            )
        
        self._portfolio  = TargetPortfolio(
            position = self._position,
            value = self.portfolio_initial_value,
            price = self._get_price()
        )
        
        self.historical_info = History(max_size= len(self.df))
        self.historical_info.set(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index =self.positions.index(self._position),
            position = self._position,
            real_position = self._position,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            portfolio_distribution = self._portfolio.get_portfolio_distribution(),
            reward = 0,
        )

        return self._get_obs(), self.historical_info[0]

    def render(self):
        pass

    def _trade(self, position, price = None):
        self._portfolio.trade_to_position(
            position, 
            price = self._get_price() if price is None else price, 
            trading_fees = self.trading_fees
        )
        self._position = position
        return

    def _take_action(self, position):
        if position != self._position:
            self._trade(position)
    
    def _take_action_order_limit(self):
        if len(self._limit_orders) > 0:
            ticker = self._get_ticker()
            for position, params in self._limit_orders.items():
                if position != self._position and params['limit'] <= ticker["high"] and params['limit'] >= ticker["low"]:
                    self._trade(position, price= params['limit'])
                    if not params['persistent']: del self._limit_orders[position]


    
    def add_limit_order(self, position, limit, persistent = False):
        self._limit_orders[position] = {
            'limit' : limit,
            'persistent': persistent
        }
    
    def step(self, continuous_action = None):
    
        if continuous_action is not None: 
            
            if continuous_action > 0.01:
                position_index = 2
        
            elif continuous_action < -0.01:
                position_index = 0
            
            else:   
                position_index = 1
                
            self._take_action(self.positions[position_index])
            
        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate= self.borrow_interest_rate)
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if isinstance(self.max_episode_duration,int) and self._step >= self.max_episode_duration - 1:
            truncated = True

        self.historical_info.add(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            position_index =position_index,
            position = self._position,
            real_position = self._portfolio.real_position(price),
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = portfolio_value,
            portfolio_distribution = portfolio_distribution, 
            reward = 0
        )
        if not done:
            reward = self.reward_function(self.historical_info, self.initial_position*2)
            self.historical_info["reward", -1] = reward

        if done or truncated:
            self.calculate_metrics()
            self.log()
        return self._get_obs(),  self.historical_info["reward", -1], done, truncated, self.historical_info[-1]

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })
    def calculate_metrics(self):
        self.results_metrics = {
            "Market Return" : f"{100*(self.historical_info['data_close', -1] / self.historical_info['data_close', 0] -1):5.2f}%",
            "Portfolio Return" : f"{100*(self.historical_info['portfolio_valuation', -1] / self.historical_info['portfolio_valuation', 0] -1):5.2f}%",
        }

        for metric in self.log_metrics:
            self.results_metrics[metric['name']] = metric['function'](self.historical_info)
    def get_metrics(self):
        return self.results_metrics
    def log(self):
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

    def save_for_render(self, dir = "render_logs"):
        assert "open" in self.df and "high" in self.df and "low" in self.df and "close" in self.df, "Your DataFrame needs to contain columns : open, high, low, close to render !"
        columns = list(set(self.historical_info.columns) - set([f"date_{col}" for col in self._info_columns]))
        history_df = pd.DataFrame(
            self.historical_info[columns], columns= columns
        )
        history_df.set_index("date", inplace= True)
        history_df.sort_index(inplace = True)
        render_df = self.df.join(history_df, how = "inner")
        
        if not os.path.exists(dir):os.makedirs(dir)
        render_df.to_pickle(f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")




#Env option 3
class ForexTradingEnv(gym.Env):
    def __init__(self, data, take_profit_position_range=(0.10, 0.80), stop_loss_position_range=(0.00, 0.15), max_stop_loss_position=0.30):
        super(ForexTradingEnv, self).__init__()

        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0 
        self.position_open = 0
        self.num_trades = 0
        self.profit_loss = 0
        self.max_stop_loss_position = max_stop_loss_position

        self.take_profit_position_range = take_profit_position_range
        self.stop_loss_position_range = stop_loss_position_range

        self.action_space = gym.spaces.Discrete(3)

        #State and Observations
        n_features = len(self.data.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(n_features,))

        self.episode_returns = []
        self.cumulative_returns = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_returns = []

        self.overall_rewards = 0


    def step(self, action):
        self.current_step += 1
        done = False

        reward = 0
        trade_outcome = 0
        trade_within_range = False

        if action == 1:  #Buy
            if self.position == 0:   #Only buy if not in a trade
                position_size = 0.05*self.initial_balance

                self.position_open = self.data.loc[self.current_step, 'open']
                self.position = 1
                self.num_trades += 1
                print(f"Opened trade at step {self.current_step} with position size: {position_size:.2f}")
        
        elif action == 2: #Close trade
            if self.position == 1: #Only close a position if currently open
                position_close = self.data.loc[self.current_step, 'open']
                trade_outcome = position_close - self.position_open        # $$$$$$$$$$$$$ When now shorting, use abs()
                self.profit_loss += np.round(position_close - self.position_open, 5)*1e4 * 1  #Mini lot (10 000 units) = $1 per pip
                self.position = 0
                self.position_open = 0
                print(f"Closed trade at step {self.current_step}")
                print(f"----------------------> Trade Outcome: {trade_outcome}")

                self.balance += self.profit_loss

                #Check for take profit and stop loss               $$$$$$ Needs attention
                if trade_outcome > 0:
                    if trade_outcome >= self.take_profit_position_range[0] * self.position:
                        reward = 1  #Trade reached take profit
                        self.winning_trades += 1 #+1 winning trades count
                        if trade_outcome <= self.take_profit_position_range[1] * self.position and trade_outcome >= self.take_profit_position_range[0] * self.position:
                            print("Trade reached take profit")

                elif trade_outcome < 0:
                    if abs(trade_outcome) >= self.max_stop_loss_position * self.position:
                        reward = -1.7  #Trade reached the maximum stop loss
                        self.losing_trades += 1 #+1 losing trades count
                        print("Trade reached the maximum stop loss")
                    
                    elif abs(trade_outcome) >= self.stop_loss_position_range[0] * self.position:
                        reward = -1 #Trade reached the stop loss
                        self.losing_trades += 1 #+1 losing trades count
                        print("Trade reached stop loss") 
                    
                    else:
                        #No reward (positive and negative) if the trade closed with no loss
                        reward = 0
                        print("Trade closed with no loss")

        if trade_outcome > 0:
            reward += 0.5
        print(f"Reward: {reward}")

        # reward for this episode
        self.episode_returns.append(reward)

        # cummulative returns 
        self.cumulative_returns += reward


        #Daily returns for daily 
        self.daily_returns.append(reward)

        self.overall_rewards += reward

        if self.current_step >= self.n_steps:
            done = True

        if self.balance <= self.initial_balance*0.85:   #### if we get to 15 % of initial balance 
            done = True

        next_state = self.get_observation()
        return next_state, reward, done
            
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.position_open = 0
        self.episode_returns = []
        episode_return = np.sum(self.episode_returns)
        self.cumulative_returns += episode_return
        self.current_week_start = 0
        return self.get_observation()
            
    def render(self, mode = 'human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Open Position: {self.position}")
            print(f"Trades: {self.num_trades} | Profit/Loss: {self.profit_loss:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Winning Trades: {self.winning_trades} | Losing Trades: {self.losing_trades}")
            print(f"Overall Rewards: {self.overall_rewards:.2f}")
            print(self.data.loc[self.current_step])

    def close(self):
        pass

    def get_observation(self):
        obs = self.data.loc[self.current_step, list(self.data.columns)].values.astype(np.float32)
        return obs / obs.max() 
