import numpy as np
import pandas as pd
from environments import TradingEnv
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO


from IPython.display import clear_output
import time


def data_func(GU_daily_path, GU_weekly_path, GU_month_path, DXY_daily_path, daily_only = True):

    GU_daily = pd.read_csv(GU_daily_path)
    GU_daily.drop(['Adj Close','Volume'],inplace=True, axis=1)
    GU_daily['Date'] = pd.to_datetime(GU_daily['Date'])
    GU_daily.columns = [x.lower() for x in list(GU_daily.columns)]

    GU_daily['day'] = GU_daily['date'].dt.dayofweek + 1
    GU_daily['week'] = GU_daily['date'].dt.isocalendar().week
    GU_daily['month'] = GU_daily['date'].dt.month
    GU_daily['year'] = GU_daily['date'].dt.year

    GU_daily.ffill(inplace=True)

    if daily_only:
       GU_daily.set_index('date', inplace=True)
       return GU_daily

    else:

       GU_weekly = pd.read_csv(GU_weekly_path)
       GU_monthly = pd.read_csv(GU_month_path)
       DXY_daily = pd.read_csv(DXY_daily_path)


       GU_weekly.drop(['Adj Close','Volume'],inplace=True, axis=1)
       GU_monthly.drop(['Adj Close','Volume'],inplace=True, axis=1)
       DXY_daily.drop(['Adj Close','Volume'],inplace=True, axis=1)


       GU_weekly['Date'] = pd.to_datetime(GU_weekly['Date'])
       GU_monthly['Date'] = pd.to_datetime(GU_monthly['Date'])
       DXY_daily['Date'] = pd.to_datetime(DXY_daily['Date'])


       GU_weekly.columns = [x.lower() for x in list(GU_weekly.columns)]
       GU_monthly.columns = [x.lower() for x in list(GU_monthly.columns)]
       DXY_daily.columns = [x.lower() for x in list(DXY_daily.columns)]


       GU_weekly['week'] = GU_weekly['date'].dt.isocalendar().week
       GU_weekly['month'] = GU_weekly['date'].dt.month
       GU_weekly['year'] = GU_weekly['date'].dt.year

       GU_monthly['month'] = GU_monthly['date'].dt.month
       GU_monthly['year'] = GU_monthly['date'].dt.year


       merged_GU = pd.merge(GU_daily, GU_weekly, on=['week', 'month', 'year'], suffixes=('_daily', '_weekly'))

       merged_GU.drop(['date_weekly'], inplace= True, axis = 1)

       merged_GU = pd.merge(merged_GU, GU_monthly, on=['month', 'year'], suffixes=('dw', '_monthly'))

       cols = []

       for i, j in enumerate (merged_GU.columns):
          if i < 5:
              cols.append(j.split('_')[0])
          elif 4 < i < 13:
              cols.append(j)
          else:
              cols.append(j + '_' +'monthly')

       merged_GU.columns = cols

       merged_GU.drop(['date_monthly', 'year'], inplace= True, axis = 1)


       DXY_daily['day'] = DXY_daily['date'].dt.dayofweek + 1
       DXY_daily['week'] = DXY_daily['date'].dt.isocalendar().week
       DXY_daily['month'] = DXY_daily['date'].dt.month
       DXY_daily['year'] = DXY_daily['date'].dt.year

       DXY_daily.drop(DXY_daily[DXY_daily['day']==7].index,axis=0, inplace=True)

       DXY_daily.ffill(inplace=True)

       merged_df = pd.merge(GU_daily, DXY_daily, on=['day', 'week', 'month', 'year'], suffixes=('_GU', '_DXY'))

       merged_df.drop(['date_DXY', 'year'], inplace=True, axis=1)


       cols = []

       for i, j in enumerate (merged_df.columns):
           if i < 5:
               cols.append(j.split('_')[0])
           else:
               cols.append(j)

       merged_df.columns = cols
        
       GU_daily.set_index('date', inplace=True)    
       GU_weekly.set_index('date', inplace=True)
       GU_monthly.set_index('date', inplace=True)
       DXY_daily.set_index('date', inplace=True)
       merged_GU.set_index('date', inplace=True)
       merged_df.set_index('date', inplace=True)
       
       GU_daily.drop('year', axis=1, inplace=True)
       GU_weekly.drop('year', axis=1, inplace=True)
       GU_monthly.drop('year', axis=1, inplace=True)
       DXY_daily.drop('year', axis=1, inplace=True)
       
       return GU_daily, GU_weekly, GU_monthly, DXY_daily, merged_GU, merged_df
       
def state_space(df):
  df = df.copy()
  
  split_date = '2021-06-30'

  #df['daily_open'] = df['open']
  #df['daily_high'] = df['high']
  #df['daily_low'] = df['low']
  df['daily_close'] = df['close']

  train_df = df.copy()[:split_date]
  test_df = df.copy()[split_date:]

  cols = []

  for i, j in enumerate(train_df):
      if i>3:
          cols.append(j+'_'+'feature')
      else:
          cols.append(j)

  train_df.columns = cols
  test_df.columns = cols
  return train_df, test_df
  
    
def simulator_descrete(model, test_df, env):

  # Run an episode until it ends :
  tot_episode_reward = 0
  episode_reward_tracker = []
  current_bal = 100000
  tot_episode_profit = 0
  episode_profit_tracker = []
  balance_tracker = []
  position_tracker = []

  done, truncated = False, False
  observation, info = env.reset()
  while not done and not truncated:
     # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
      action,_ = model.predict(observation, deterministic=False)
      observation, reward, done, truncated, info = env.step(action)

      env.render()
      time.sleep(0.1)
      clear_output(wait=True)

    #More metrics
      balance_tracker.append(current_bal)
      tot_episode_reward += reward
      episode_reward_tracker.append(reward)

      profit = info['portfolio_valuation'] - current_bal
      tot_episode_profit += profit
      episode_profit_tracker.append(profit)
      current_bal = info['portfolio_valuation']

      position_tracker.append(info['position'])

  fig, axs = plt.subplots(nrows=5, figsize=(8,10))

  sns.lineplot(x=test_df.index[:-2], y=test_df['close'][:-2], ax=axs[0])
  axs[0].set_title(f'Test GBPUSD daily close from {test_df.index[0]} to {test_df.index[-1]}.')
  axs[0].set_xlabel('Date')
  axs[0].set_ylabel('GBPUSD close price [$/Pound]')

  sns.lineplot(x=test_df.index[:-2], y=balance_tracker, ax=axs[1])
  axs[1].set_title('Balance during the episode.')
  axs[1].set_xlabel('Date')
  axs[1].set_ylabel('Balance [$]')

  sns.scatterplot(x=test_df.index[:-2], y=position_tracker, ax=axs[2])
  axs[2].set_title('Position tracker during the episode.')
  axs[2].set_xlabel('Date')
  axs[2].set_ylabel('Position')

  sns.lineplot(x=test_df.index[:-2], y=episode_profit_tracker, ax=axs[3], ls='--', color='g', size=0.5, legend=False)
  sns.scatterplot(x=test_df.index[:-2], y=episode_profit_tracker, ax=axs[3], hue=[x>=0 for x in episode_profit_tracker], legend=False)
  axs[3].set_title('Profit tracker for the entire test episode.')
  axs[3].set_xlabel('Date')
  axs[3].set_ylabel('Profit [$]')

  sns.scatterplot(x=test_df.index[:-2], y=episode_reward_tracker, ax=axs[4], hue=[x>=0 for x in episode_reward_tracker], legend=False)
  axs[4].set_title('Rewards during the entire test episode.')
  axs[4].set_xlabel('Date')
  axs[4].set_ylabel('Reward')

  plt.tight_layout()
  plt.show()





# Function to simulate trading using the trained model and the provided DataFrame
def Visual_Simulator(Data, Model:'dqn', Strategy:'ewm'):

  # Setting the risk level for trading positions
  risk = 0.5

  if Strategy == 'ewm':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculating the 20-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_20'
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    # Calculating the 9-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_9'

    # Backfilling any missing values in the DataFrame
    df.bfill(inplace=True)

    # Dropping the 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_ewm_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_ewm_best")

  elif Strategy == 'macd':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculating the 12-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_12'
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    # Calculating the 26-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_26'
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Calculating the Moving Average Convergence Divergence (MACD) by subtracting EMA_26 from EMA_12
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    # Calculating the 9-period EMA of the MACD and storing it in a new column 'MACD_9'
    df['MACD_9'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Dropping the columns from the DataFrame as they are no longer needed
    df.drop(['EMA_12', 'EMA_26', 'open', 'high', 'low'], inplace=True, axis=1)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_macd_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_macd_best")


  elif Strategy == 'bb':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculating the upper Bollinger Band
    df['BB_upper'] = df['close'].rolling(window=10).mean() + 2 * df['close'].rolling(window=10).std()
    # Calculating the lower Bollinger Band
    df['BB_lower'] = df['close'].rolling(window=10).mean() - 2 * df['close'].rolling(window=10).std()

    # Dropping rows where the 'BB_upper' column has NaN values
    df.drop(df[df['BB_upper'].isna()].index, axis=0, inplace=True)
    # Dropping rows where the 'BB_lower' column has NaN values
    df.drop(df[df['BB_lower'].isna()].index, axis=0, inplace=True)
    # Dropping the 'open', 'high', and 'low' columns from the DataFrame as they are no longer needed
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_bb_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_bb_best")

  elif Strategy == 'rsi':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculate the difference between consecutive 'close' prices
    delta = df['close'].diff()
    # Calculate gains (positive differences) and losses (negative differences)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and loss over a 14-period window
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the Relative Strength Index (RSI)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Dropping the 'open', 'high', and 'low' columns from the DataFrame as they are no longer needed
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_rsi_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_rsi_best")

  elif Strategy == 'SR_1':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculate Tenkan-sen (Conversion Line)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    # Calculate Kijun-sen (Base Line)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    # Calculate Senkou Span A (Leading Span A)
    df['Senkou_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    # Calculate Senkou Span B (Leading Span B)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)

    # Calculate Chikou Span (Lagging Span)
    df['Chikou_span'] = df['close'].shift(-26)

    # Drop the 'Tenkan_sen', 'Kijun_sen', 'Chikou_span', 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['Tenkan_sen', 'Kijun_sen','Chikou_span', 'open', 'high', 'low'], axis=1, inplace=True)

    # Drop rows where the 'Senkou_A' and 'Senkou_B' columns have NaN values
    df.drop(df[df['Senkou_A'].isna()].index, axis=0, inplace=True)
    df.drop(df[df['Senkou_B'].isna()].index, axis=0, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_S&R_ichimoko_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_S&R_ichimoko_best")


  elif Strategy == 'SR_2':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    my_levels = [1.84, 1.76, 1.974, 1.954, 1.9, 2.079, 2.013, 1.598, 1.63, 1.551, 1.33, 1.629, 1.376, 1.081, 1.168, 1.45, 1.7, 1.255, 2.13, 1.499, 1.256]

    # Iterating over the filtered levels and their indices
    for x,y in enumerate(my_levels):
    # Creating a new column in df1 for each level, named by the index
    # Each new column is filled with the corresponding level value 'y'
      df[str(x)] = df['open'].copy().apply(lambda c: y)

    # Drop the 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_S&R_trendln_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_S&R_trendln_best")



  elif Strategy == 'lstm':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Load LSTM hidden features from saved CSV
    lstm_df = pd.read_csv('/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/data/lstm_hidden_features', index_col=0)

    # Align the index of the LSTM DataFrame with the GU_daily DataFrame, starting from the 30th index
    lstm_df.index = df.index[30:]

    # Concatenate GBPUSD daily dataframe with the LSTM hidden features
    df = pd.concat([df[30:], lstm_df], axis=1)

    # Drop the 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_lstm_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_lstm_best")

  elif Strategy == 'mtf_1':
    # Make copy of merged multi-timeframe GBPUSD dataframe
    df = Data[4].copy()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_9'] = df['close_weekly'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_20'] = df['close_weekly'].ewm(span=20, adjust=False).mean()

    # Drop the columns from the DataFrame
    df.drop(['open_weekly', 'high_weekly', 'low_weekly', 'open_monthly', 'high_monthly', 'low_monthly', 'close_monthly', 'open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_mtf_dw_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_mtf_dw_best")


  elif Strategy == 'mtf_2':
    # Make copy of merged multi-timeframe GBPUSD dataframe
    df = Data[4].copy()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_9'] = df['close_weekly'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_20'] = df['close_weekly'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the monthly 'close' prices
    df['monthly_EMA_9'] = df['close_monthly'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the monthly 'close' prices
    df['monthly_EMA_20'] = df['close_monthly'].ewm(span=20, adjust=False).mean()

    # Drop the specified columns from the DataFrame
    df.drop(['open_weekly', 'high_weekly', 'low_weekly', 'open_monthly', 'high_monthly', 'low_monthly', 'open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_mtf_dwm_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_mtf_dwm_best")


  elif Strategy == 'dxy':
    #  Make copy of merged GBPUSD dataframe
    df = Data[-1].copy()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close_DXY' prices
    df['dxy_EMA_9'] = df['close_DXY'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close_DXY' prices
    df['dxy_EMA_20'] = df['close_DXY'].ewm(span=20, adjust=False).mean()

    # Backfill missing values in the DataFrame
    df.bfill(inplace=True)

    # Drop the specified columns from the DataFrame
    df.drop(['open_DXY', 'high_DXY', 'low_DXY', 'close_DXY', 'open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_dxy_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_dxy_best")



  # Extracting the testing data from the state space
  test_df = state_space(df)[1]

  # Creating the trading environment for testing with specified parameters
  env = TradingEnv(df=test_df, positions = [-risk, 0, risk], portfolio_initial_value = 100000, initial_position= risk, max_episode_duration=len(test_df)-1)

  # Run an episode until it ends :
  tot_episode_reward = 0
  episode_reward_tracker = []
  current_bal = 100000
  tot_episode_profit = 0
  episode_profit_tracker = []
  balance_tracker = []
  position_tracker = []

  done, truncated = False, False
  observation, info = env.reset()
  while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
      action,_ = model.predict(observation, deterministic=False)
      observation, reward, done, truncated, info = env.step(action)

      env.render()
      time.sleep(0.1)
      clear_output(wait=True)

    #More metrics
      balance_tracker.append(current_bal)
      tot_episode_reward += reward
      episode_reward_tracker.append(reward)

      profit = info['portfolio_valuation'] - current_bal
      tot_episode_profit += profit
      episode_profit_tracker.append(profit)
      current_bal = info['portfolio_valuation']

      position_tracker.append(info['position'])

  fig, axs = plt.subplots(nrows=5, figsize=(8,10))

  sns.lineplot(x=test_df.index[:-2], y=test_df['close'][:-2], ax=axs[0])
  axs[0].set_title(f'Test GBPUSD daily close from {test_df.index[0]} to {test_df.index[-1]}.')
  axs[0].set_xlabel('Date')
  axs[0].set_ylabel('GBPUSD close price [$/Pound]')

  sns.lineplot(x=test_df.index[:-2], y=balance_tracker, ax=axs[1])
  axs[1].set_title('Balance during the episode.')
  axs[1].set_xlabel('Date')
  axs[1].set_ylabel('Balance [$]')

  sns.scatterplot(x=test_df.index[:-2], y=position_tracker, ax=axs[2])
  axs[2].set_title('Position tracker during the episode.')
  axs[2].set_xlabel('Date')
  axs[2].set_ylabel('Position')

  sns.lineplot(x=test_df.index[:-2], y=episode_profit_tracker, ax=axs[3], ls='--', color='g', size=0.5, legend=False)
  sns.scatterplot(x=test_df.index[:-2], y=episode_profit_tracker, ax=axs[3], hue=[x>=0 for x in episode_profit_tracker], legend=False)
  axs[3].set_title('Profit tracker for the entire test episode.')
  axs[3].set_xlabel('Date')
  axs[3].set_ylabel('Profit [$]')

  sns.scatterplot(x=test_df.index[:-2], y=episode_reward_tracker, ax=axs[4], hue=[x>=0 for x in episode_reward_tracker], legend=False)
  axs[4].set_title('Rewards during the entire test episode.')
  axs[4].set_xlabel('Date')
  axs[4].set_ylabel('Reward')

  plt.tight_layout()
  plt.show()

    # Running the simulation and returning the results
  return    
    



#Function to simulate trading using the trained model and the provided DataFrame
def Universal_Simulator(Data, Model:'dqn', Strategy:'ewm'):

  # Setting the risk level for trading positions
  risk = 0.5

  if Strategy == 'ewm':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculating the 20-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_20'
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    # Calculating the 9-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_9'

    # Backfilling any missing values in the DataFrame
    df.bfill(inplace=True)

    # Dropping the 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_ewm_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_ewm_best")

  elif Strategy == 'macd':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculating the 12-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_12'
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    # Calculating the 26-period Exponential Moving Average (EMA) of the 'close' prices and storing it in a new column 'EMA_26'
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # Calculating the Moving Average Convergence Divergence (MACD) by subtracting EMA_26 from EMA_12
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    # Calculating the 9-period EMA of the MACD and storing it in a new column 'MACD_9'
    df['MACD_9'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Dropping the columns from the DataFrame as they are no longer needed
    df.drop(['EMA_12', 'EMA_26', 'open', 'high', 'low'], inplace=True, axis=1)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_macd_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_macd_best")


  elif Strategy == 'bb':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculating the upper Bollinger Band
    df['BB_upper'] = df['close'].rolling(window=10).mean() + 2 * df['close'].rolling(window=10).std()
    # Calculating the lower Bollinger Band
    df['BB_lower'] = df['close'].rolling(window=10).mean() - 2 * df['close'].rolling(window=10).std()

    # Dropping rows where the 'BB_upper' column has NaN values
    df.drop(df[df['BB_upper'].isna()].index, axis=0, inplace=True)
    # Dropping rows where the 'BB_lower' column has NaN values
    df.drop(df[df['BB_lower'].isna()].index, axis=0, inplace=True)
    # Dropping the 'open', 'high', and 'low' columns from the DataFrame as they are no longer needed
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_bb_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_bb_best")

  elif Strategy == 'rsi':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculate the difference between consecutive 'close' prices
    delta = df['close'].diff()
    # Calculate gains (positive differences) and losses (negative differences)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and loss over a 14-period window
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the Relative Strength Index (RSI)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Dropping the 'open', 'high', and 'low' columns from the DataFrame as they are no longer needed
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_rsi_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_rsi_best")

  elif Strategy == 'SR_1':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Calculate Tenkan-sen (Conversion Line)
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    # Calculate Kijun-sen (Base Line)
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    # Calculate Senkou Span A (Leading Span A)
    df['Senkou_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    # Calculate Senkou Span B (Leading Span B)
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)

    # Calculate Chikou Span (Lagging Span)
    df['Chikou_span'] = df['close'].shift(-26)

    # Drop the 'Tenkan_sen', 'Kijun_sen', 'Chikou_span', 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['Tenkan_sen', 'Kijun_sen','Chikou_span', 'open', 'high', 'low'], axis=1, inplace=True)

    # Drop rows where the 'Senkou_A' and 'Senkou_B' columns have NaN values
    df.drop(df[df['Senkou_A'].isna()].index, axis=0, inplace=True)
    df.drop(df[df['Senkou_B'].isna()].index, axis=0, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_S&R_ichimoko_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_S&R_ichimoko_best")


  elif Strategy == 'SR_2':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    my_levels = [1.84, 1.76, 1.974, 1.954, 1.9, 2.079, 2.013, 1.598, 1.63, 1.551, 1.33, 1.629, 1.376, 1.081, 1.168, 1.45, 1.7, 1.255, 2.13, 1.499, 1.256]

    # Iterating over the filtered levels and their indices
    for x,y in enumerate(my_levels):
    # Creating a new column in df1 for each level, named by the index
    # Each new column is filled with the corresponding level value 'y'
      df[str(x)] = df['open'].copy().apply(lambda c: y)

    # Drop the 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_S&R_trendln_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_S&R_trendln_best")



  elif Strategy == 'lstm':
    # Make copy of GBPUSD daily
    df = Data[0].copy()

    # Load LSTM hidden features from saved CSV
    lstm_df = pd.read_csv('/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/data/lstm_hidden_features', index_col=0)

    # Align the index of the LSTM DataFrame with the GU_daily DataFrame, starting from the 30th index
    lstm_df.index = df.index[30:]

    # Concatenate GBPUSD daily dataframe with the LSTM hidden features
    df = pd.concat([df[30:], lstm_df], axis=1)

    # Drop the 'open', 'high', and 'low' columns from the DataFrame
    df.drop(['open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_lstm_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_lstm_best")

  elif Strategy == 'mtf_1':
    # Make copy of merged multi-timeframe GBPUSD dataframe
    df = Data[4].copy()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_9'] = df['close_weekly'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_20'] = df['close_weekly'].ewm(span=20, adjust=False).mean()

    # Drop the columns from the DataFrame
    df.drop(['open_weekly', 'high_weekly', 'low_weekly', 'open_monthly', 'high_monthly', 'low_monthly', 'close_monthly', 'open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_mtf_dw_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_mtf_dw_best")


  elif Strategy == 'mtf_2':
    # Make copy of merged multi-timeframe GBPUSD dataframe
    df = Data[4].copy()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_9'] = df['close_weekly'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the weekly 'close' prices
    df['weekly_EMA_20'] = df['close_weekly'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the monthly 'close' prices
    df['monthly_EMA_9'] = df['close_monthly'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the monthly 'close' prices
    df['monthly_EMA_20'] = df['close_monthly'].ewm(span=20, adjust=False).mean()

    # Drop the specified columns from the DataFrame
    df.drop(['open_weekly', 'high_weekly', 'low_weekly', 'open_monthly', 'high_monthly', 'low_monthly', 'open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_mtf_dwm_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_mtf_dwm_best")


  elif Strategy == 'dxy':
    #  Make copy of merged GBPUSD dataframe
    df = Data[-1].copy()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close' prices
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Calculate the 9-period Exponential Moving Average (EMA) for the 'close_DXY' prices
    df['dxy_EMA_9'] = df['close_DXY'].ewm(span=9, adjust=False).mean()
    # Calculate the 20-period Exponential Moving Average (EMA) for the 'close_DXY' prices
    df['dxy_EMA_20'] = df['close_DXY'].ewm(span=20, adjust=False).mean()

    # Backfill missing values in the DataFrame
    df.bfill(inplace=True)

    # Drop the specified columns from the DataFrame
    df.drop(['open_DXY', 'high_DXY', 'low_DXY', 'close_DXY', 'open', 'high', 'low'], axis=1, inplace=True)

    if Model == 'dqn':
      model = DQN.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/dqn_dxy_best")

    else:
      model = PPO.load("/content/drive/MyDrive/MScDataScienceArtificialIntelligence/3rdSemester/Dissertation/Build/models/ppo_dxy_best")



  # Extracting the testing data from the state space
  test_df = state_space(df)[1]

  # Creating the trading environment for testing with specified parameters
  env = TradingEnv(df=test_df, positions = [-risk, 0, risk], portfolio_initial_value = 100000, initial_position= risk, max_episode_duration=len(test_df)-1)

  # Run an episode until it ends :
  tot_episode_reward = 0
  episode_reward_tracker = []
  current_bal = 100000
  tot_episode_profit = 0
  episode_profit_tracker = []
  balance_tracker = []
  position_tracker = []

  done, truncated = False, False
  observation, info = env.reset()
  while not done and not truncated:
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
      action,_ = model.predict(observation, deterministic=False)
      observation, reward, done, truncated, info = env.step(action)

      env.render()
      time.sleep(0.1)
      clear_output(wait=True)

    #More metrics
      balance_tracker.append(current_bal)
      episode_reward_tracker.append(reward)

      profit = info['portfolio_valuation'] - current_bal
      current_bal = info['portfolio_valuation']

  episode_reward_tracker = np.array(episode_reward_tracker)
  balance_tracker = np.array(balance_tracker)

  sim_df = pd.DataFrame({'episode_reward_tracker': episode_reward_tracker, 'balance_tracker': balance_tracker}, index = test_df.index[:-2])

  return sim_df