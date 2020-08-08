import numpy as np
import utils

class Agent:
    # 에이전트가 구성하는 값 개수
    # 주식 보유 비율, 포트폴리오 가치 비율
    STATE_DIM = 2

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.0025
    TRADING_TAX = 0.003

    # actions
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    # 인공 신경망에서 확률을 구할 행동들
    # 살건지 안살건지만 구하면 됨 홀딩은 확률계산 불필요
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIOONS = len(ACTIONS) # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=10, delayed_reward_threshold=.05):

        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매단위, 최대 매매 단위
        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        # 지연보상 임계치
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성

        self.initial_balance = 0 # 초기 자본금
        self.balance = 0 # 현재 자본금
        self.num_stocks = 0 
        
        # 포트폴리오 가치 : balance + num_stocks * current_stock_price
        self.prtfolio_value = 0
        self.base_portfolio_value = 0 # 직전 학습 시점의 포트폴리오
        self.num_buy = 0 # 매수 횟수
        self.num_sell = 0 # 매도 횟수
        self.num_hold = 0 # 홀딩 횟수
        self.immediate_reward = 0 # 즉시 보상
        self.profitloss = 0 # 현재 손익
        self.base_profitloss = 0 # 직전 지연 보상 이후 손익
        self.exploration_base = 0 # 참험 행동 결정 기준

        # Agent 클래스의 상태 
        self.ratio_hold = 0 # 주식 보유 비율
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (self.portfolio_value / self.base_portfolio_value)

        return (self.ratio_hold, self.ratio_portfolio_value)

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0

        pred = pred_policy
        if pred is None:
            pred = pred_value
        
        if pred is None:
            epsilon = 1
        else:
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
        
        # 탐험결정 랜덤으로 할거냐 예측값으로 할거냐
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIOONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)
        
        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])
        
        return action, confidence, exploration


    def vaildate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)), self.max_trading_unit - self.min_trading_unit), 0)

        return self.min_trading_unit + added_trading
    

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit)

            # 보유현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(
                            self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                            self.max_trading_unit
                            ),
                            self.min_trading_unit
                            )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit

            if invest_amount > 0:
                self.balance -= invest_amount # 보유현금을 갱신
                self.num_stocks += trading_unit # 보유 주식 수 갱신
                self.num_buy += 1 # 매도 횟수 증가
        
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)

            # 매도
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit

            if invest_amount > 0:
                self.balance += invest_amount
                self.num_stocks -= trading_unit
                self.num_sell += 1
        
        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1
        
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = ((self.portfolio_value - self.initial_balance) / self.initial_balance)

        # 즉시보상 = 수익률
        self.immediate_reward = self.profitloss

        # 지연보상 = 익절 손절 기준
        delayed_reward = 0
        self.base_profitloss = ((self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < self.delayed_reward_threshold:
            # 목표 수익률을 달성하여 기존 포트폴리오 가치 갱신
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0
        
        return self.immediate_reward, delayed_reward
        
            
















