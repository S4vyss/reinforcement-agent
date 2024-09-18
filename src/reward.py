import cProfile
import io
import pstats
import numpy as np
import pandas as pd
from fe import train, valid

def calculate_bollinger_bands(prices, window=20, num_std=2):
    if len(prices) < window:
        sma = [np.nan] * len(prices)
        upper_band = [np.nan] * len(prices)
        lower_band = [np.nan] * len(prices)
    else:
        series = pd.Series(prices[-window:])
        windows = series.rolling(window)
        sma = windows.mean().tolist()[window - 1:]
        std = windows.std().tolist()[window - 1:]

        upper_band = [sma_val + num_std *
                      std_val for sma_val, std_val in zip(sma, std)]
        lower_band = [sma_val - num_std *
                      std_val for sma_val, std_val in zip(sma, std)]

    return sma, upper_band, lower_band


def calculate_reward(history, training=True):
    current_step = history[-1]
    # save_history.append(history)
    # Extract closing prices from the history object
    closing_prices = train['close'][:current_step['step']
                                    ] if training else valid['close'][:current_step['step']]
    reward = current_step['reward']
    position = np.array(current_step['position'])

    sma, upper_band, lower_band = calculate_bollinger_bands(closing_prices)
    if np.isnan(upper_band[-1]) or np.isnan(lower_band[-1]):
        pass
    else:
        current_price = current_step['data_close']

        long_condition = (position > 0) & (current_price > upper_band[-1])
        short_condition = (position < 0) & (current_price < lower_band[-1])
        reward = np.where(long_condition, reward * 1.5, reward)
        reward = np.where(short_condition, reward * 1.5, reward)

        long_condition = (position > 0) & (current_price < lower_band[-1])
        short_condition = (position < 0) & (current_price > upper_band[-1])
        reward = np.where(long_condition, reward * 0.5, reward)
        reward = np.where(short_condition, reward * 0.5, reward)

    portfolio_growth = (history['portfolio_valuation'][-1] -
                        history['portfolio_valuation'][-2]) / history['portfolio_valuation'][-2]
    reward += portfolio_growth

    market_return = (current_step['data_close'] -
                     current_step['data_open']) / current_step['data_open']

    if market_return < 0 and position < 0:
        reward += 1  # Reward for short position in a declining market
    elif market_return > 0 and position > 0:
        reward += 1  # Reward for long position in a rising market
    elif market_return < 0 and position > 0:
        reward -= 1  # Penalize for long position in a declining market
    elif market_return > 0 and position < 0:
        reward -= 1  # Penalize for short position in a rising market

    if np.isnan(reward) or np.isinf(reward):
        raise ValueError("Reward function produced NaN or infinity values")

    return reward


def profile_function(history):
    pr = cProfile.Profile()
    pr.enable()

    reward = calculate_reward(history)

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.TIME
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def reward_function(history):
    current_step = history[-1]
    position = np.array(current_step['position'])
    portfolio_growth = history["portfolio_valuation", -
                               1] / history["portfolio_valuation", -2]
    reward = np.log(portfolio_growth)
    if history["portfolio_valuation", -1] < 200:
        reward -= np.log(history["portfolio_valuation", -1])

    market_return = (current_step['data_close'] -
                     current_step['data_open']) / current_step['data_open']

    if market_return < 0 and position < 0:
        # Reward for short position in a declining market
        reward += abs(market_return)
    elif market_return > 0 and position > 0 or position == 0:
        reward += market_return  # Reward for long position in a rising market
    elif market_return < 0 and position > 0 or position == 0:
        # Penalize for long position in a declining market
        reward -= abs(market_return)
    elif market_return > 0 and position < 0:
        reward -= market_return  # Penalize for short position in a rising market

    return reward
