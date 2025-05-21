import numpy as np
import pandas as pd
from gym import Env, spaces

class StockTradingEnv(Env):
    def __init__(self, stock_data, initial_balance=10000):
        self.stock_data = stock_data
        self.n_step = len(stock_data)
        self.action_space = spaces.Discrete(3)  # 0=vender, 1=mantener, 2=comprar
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        row = self.stock_data.iloc[self.current_step]
        obs = np.array([
            self.balance / 10000,
            self.shares_held,
            row['close'] / row['close'],
            row['ma_10'] / row['close'],
            row['ma_50'] / row['close'],
            row['rsi'] / 100
        ])
        return obs

    def _take_action(self, action):
        self.current_price = self.stock_data.iloc[self.current_step]['close']
        if action == 2 and self.balance >= self.current_price:  # Comprar
            self.shares_held += 1
            self.balance -= self.current_price
        elif action == 0 and self.shares_held > 0:  # Vender
            self.shares_held -= 1
            self.balance += self.current_price

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        portfolio_value = self.balance + (self.shares_held * self.current_price)
        reward = (portfolio_value - 10000) / 10000
        self.done = self.current_step >= self.n_step - 1
        return self._next_observation(), reward, self.done, {}