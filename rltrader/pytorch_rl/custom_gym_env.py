
# render 는 환경에서 설정

import gym
import numpy as np
from collections import deque

MAX_ACCOUNT_BALANCE = 2147483647
LOOKBACK_WINDOW_SIZE = 20
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000

class CustomStockEnv(gym.Env):

    metadata = {'render.modes' : ['live', 'file', 'none']}


    
    def __init__(self, data):
        super(CustomStockEnv, self).__init__()

        self.data = data

        self.max_open = data['Open'].max()
        self.max_high = data['High'].max()
        self.max_close = data['Close'].max()
        self.max_low = data['Low'].max()
        self.max_volume = data['Volume'].max()

        # 보상범위 설정 0 ~ 
        self.reward_range = (-MAX_ACCOUNT_BALANCE, MAX_ACCOUNT_BALANCE)
        # 행동범위 설정 매도 매수 홀드
        self.action_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([3, 1]), dtype=np.float16)
        # 관찰 범위 설정 // 5개 특징 20+2일치
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5, LOOKBACK_WINDOW_SIZE + 2), dtype=np.float16)

    def reset(self):
        # 순이익
        self.profits = 0
        # 잔고
        self.balance = INITIAL_ACCOUNT_BALANCE
        # 순자산액 
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        # 최대 순자산액
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        # 가지고 있는 주식 수
        self.shares_held = 0
        # 원래 재산 비용? adjusted stock?
        self.cost_basis = 0
        # 주식 팔은 양
        self.total_shares_sold = 0
        # 총 팔은 금액
        self.total_sales_values = 0
        # 현재 단계
        self.current_step = 0
        # 거래한거
        self.trades = []
        
        return _next_observation()


    def step(self, action):
        self.current_step += 1
        self.delay_modifier = (self.current_step / MAX_STEPS)
        
        # 보상 = (순자산 - 기초자산) * 딜레이 
        reward = (self.net_worth - INITIAL_ACCOUNT_BALANCE) * self.delay_modifier
        done = self.net_worth <= 0 or self.current_step >= len(self.data.loc[:, 'Open'].values)
        observation = self._next_observation()
        
        return observation, reward, done, {}


    def _next_observation(self):

        self.obs_open = deque(np.zeros(20), maxlen=20)
        self.obs_high = deque(np.zeros(20), maxlen=20)
        self.obs_close = deque(np.zeros(20), maxlen=20)
        self.obs_low = deque(np.zeros(20), maxlen=20)
        self.obs_volume = deque(np.zeros(20), maxlen=20)

        self.obs_balance = deque(np.zeros(20), maxlen=20)
        self.obs_net_worth = deque(np.zeros(20), maxlen=20)
        self.obs_shares_held = deque(np.zeros(20), maxlen=20)
        self.obs_cost_basis = deque(np.zeros(20), maxlen=20)
        self.obs_total_sales_values = deque(np.zeros(20), maxlen=20)
        self.obs_profits = deque(np.zeros(20), maxlen=20)


        self.stock_open = self.data.loc[self.current_step, 'Open'].values / self.max_open 
        self.stock_high = self.data.loc[self.current_step, 'High'].values / self.max_high
        self.stock_close = self.data.loc[self.current_step, 'Close'].values / self.max_close
        self.stock_low = self.data.loc[self.current_step, 'Low'].values / self.max_low
        self.stock_volume = self.data.loc[self.current_step, 'Volume'].values / self.max_volume

        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)

        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)
        self.obs_open.append(self.stock_open)


        self.balance = self.balance / MAX_ACCOUNT_BALANCE
        self.net_worth = self.net_worth / MAX_ACCOUNT_BALANCE
        self.shares_held = self.shares_held / MAX_ACCOUNT_BALANCE
        self.cost_basis = self.cost_basis / MAX_ACCOUNT_BALANCE
        self.total_sales_values = self.total_sales_values / MAX_ACCOUNT_BALANCE
        # self.profits = 

        observation = [self.obs_open, self.obs_high, self.obs_close, self.obs_low, self.obs_volume,\
            self.obs_balance, self.obs_net_worth, self.obs_shares_held, self.obs_cost_basis, self.obs_total_sales_values, self.obs_profits]
    
        
        return observation






    def render(self, mode='live', close=False):
        pass


    
        