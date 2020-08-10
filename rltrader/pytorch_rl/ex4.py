import gym
from ex3 import Agent
from utils import plot_learning_curve
import numpy as np

import gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL



if __name__ == "__main__":
    env = gym.make('stocks-v0',
               df = STOCKS_GOOGL,
               window_size = 10,
               frame_bound = (10, 300))

    # env = gym.make('LunarLander-v2')

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01, input_dims=[10, 2], lr=0.003)
    scores, eps_history = [],[]
    n_games = 10
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        print(observation)
    #     while not done:
    #         action = agent.choose_action(observation)
    #         observation_, reward, done, info = env.step(action)
    #         score += reward
    #         agent.store_transition(observation, action, reward, observation_, done)
    #         agent.learn()
    #         observation = observation_
    #     scores.append(score)
    #     eps_history.append(agent.epsilon)

    #     avg_score = np.mean(scores[-100:])

    #     print('eps ', i, 'score %.2f' % score, 'avg score %.2f' % avg_score, 'eps %.2f' %agent.epsilon)
    
    # x = [i+1 for i in range(n_games)]
    # filename = 'lunar_lander_2020.png'
    # plot_learning_curve(x, scores, eps_history, filename)
