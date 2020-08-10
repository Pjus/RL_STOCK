import gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL


# env = gym.make('forex-v0')
# env = gym.make('stocks-v0')

custom_env = gym.make('stocks-v0',
               df = STOCKS_GOOGL,
               window_size = 10,
               frame_bound = (10, 300))

# print("env information:")
# print("> shape:", env.shape)
# print("> df.shape:", env.df.shape)
# print("> prices.shape:", env.prices.shape)
# print("> signal_features.shape:", env.signal_features.shape)
# print("> max_possible_profit:", env.max_possible_profit())

print("custom_env information:")
print("> shape:", custom_env.shape)
print("> df.shape:", custom_env.df.shape)
print("> prices.shape:", custom_env.prices.shape)
print("> signal_features.shape:", custom_env.signal_features.shape)
print("> max_possible_profit:", custom_env.max_possible_profit())


custom_env.reset()
custom_env.render()
