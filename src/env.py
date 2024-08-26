import numpy as np
import tensorflow as tf
from reward import reward_function
from tensorflow.python.framework import tensor_spec as tensor_s
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
import gymnasium as gym
from gym_trading_env.environments import TradingEnv
from fe import train
from tf_agents.environments import tf_py_environment

positions = [-1, 0, 1, -2.0, -1.0, -1.0, -0.33333333333333337, 0.33333333333333326, 1.0, 1.0, 2.0, -2.0, -1.8571428571428572, -1.7142857142857144,
             -1.5714285714285714, -1.4285714285714286, -1.2857142857142858, -
             1.1428571428571428, -1.0, -0.8571428571428572, -0.7142857142857144,
             -0.5714285714285716, -0.4285714285714286, -0.2857142857142858, -
             0.14285714285714302, 0.0, 0.1428571428571428, 0.2857142857142856,
             0.4285714285714284, 0.5714285714285712, 0.714285714285714, 0.8571428571428568, 1.0, 1.1428571428571428, 1.2857142857142856,
             1.4285714285714284, 1.5714285714285712, 1.714285714285714, 1.8571428571428568, 2.0
             ]


class GymWrapper(py_environment.PyEnvironment):
    def __init__(self, gym_env, df, window_size):
        super(GymWrapper, self).__init__()
        self._gym_env = gym_env
        self.df = df
        self.window_size = window_size
        self._action_spec = self._get_action_spec()
        self._observation_spec = self._get_observation_spec()
        self._reward_spec = self._get_reward_spec()
        self._step_type_spec = self._get_step_type_spec()
        self._discount_spec = self._get_discount_spec()
        self._info_buffer = []

    def _get_reward_spec(self):
        return tf.TensorSpec(shape=(), dtype=np.float32, name='reward')

    def _get_action_spec(self):
        action_space = self._gym_env.action_space
        if isinstance(action_space, gym.spaces.Box):
            return tensor_s.BoundedTensorSpec(
                shape=action_space.shape,
                dtype=action_space.dtype,
                minimum=action_space.low,
                maximum=action_space.high
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            return tensor_s.BoundedTensorSpec(
                shape=(),
                dtype=action_space.dtype,
                minimum=0,
                maximum=action_space.n-1
            )
        else:
            raise ValueError(
                f"Unsupported action space type: {type(action_space)}")

    def _get_observation_spec(self):
        observation_space = self._gym_env.observation_space
        return tf.TensorSpec(
            shape=observation_space.shape,
            dtype=observation_space.dtype
        )

    def _get_step_type_spec(self):
        return tf.TensorSpec(shape=(), dtype=np.int32, name='step_type')

    def _get_discount_spec(self):
        return tensor_s.BoundedTensorSpec(shape=(), dtype=np.float32, name='discount', minimum=0.0, maximum=1.0)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec

    def step_type_spec(self):
        return self._step_type_spec

    def discount_spec(self):
        return self._discount_spec

    def logging_buffer(self):
        return self._info_buffer[-2]

    def change_df(self, new_start, window_size):
        self._gym_env.set_df(
            self.df.iloc[new_start:new_start + window_size].copy())

    def _reset(self):
        # Randomly select a start point for the window
        start_index = np.random.randint(0, len(self.df) - self.window_size)
        self.change_df(start_index, self.window_size)
        print(f"New window: {start_index} - {start_index + self.window_size}\n")
        obs = self._gym_env.reset()

        # Check if obs is a tuple/list containing the observation array and info dict
        if isinstance(obs, tuple) or isinstance(obs, list):
            observation = obs[0]  # Extract the actual observation array
        else:
            observation = obs  # obs is already the observation array

        return ts.restart(observation)

    def _step(self, action):
        obs, reward, done, truncated, info = self._gym_env.step(action)

        self._info_buffer.append(info)

        if 'portfolio_valuation' in info and info['portfolio_valuation'] < 100 or done or truncated:
            print("Episode ended. Terminating episode.")
            self._gym_env.reset()
            return ts.termination(obs, reward)

        return ts.transition(obs, reward)


class MyTradingEnv(TradingEnv):
    def set_df(self, df):
        self._set_df(df)


env = MyTradingEnv(
    name="BTCUSD",
    df=train,
    positions=[-0.5, 0, 0.5],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
    trading_fees=0.01/100,  # 0.01% per stock buy / sell (Binance fees)
    # 0.0003% per timestep (one timestep = 1h here)
    borrow_interest_rate=0.0003/100,
    verbose=1,
    reward_function=lambda history: reward_function(history),
    windows=60
    )
tf_env = GymWrapper(gym_env=env, df=train, window_size=60*24)
tf_env = tf_py_environment.TFPyEnvironment(tf_env)