import gym
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v1')
goal_step = 50

# while True:
#     obs = env.reset()
#     for i in range(goal_step):
#         obs, reward, done, info = env.step(random.randrange(0, 2))
#         if done: break
#         env.render()

# reward 합을 최대화 하는 신경망 제작
# N 실행횟수 K 그 중에 뽑을 데이터 갯수 f 동작 결정 함수
def data_preparation(N, K, f, render=False):
    game_data = []

    for i in range(N):
        score = 0
        game_steps = []
        obs = env.reset()
        for step in range(goal_step):
            if render: env.render()
            action = f(obs)
            game_steps.append((obs, action))
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        game_data.append((score, game_steps))
    game_data.sort(key=lambda s: -s[0])

    traning_set = []

    for i in range(K):
        for step in game_data[i][1]:
            if step[1] == 0:
                traning_set.append((step[0], [1, 0]))
            else:
                traning_set.append((step[0], [0, 1]))

    print("{0}/{1}th score: {2}".format(K, N, game_data[K-1][0]))
    if render:
        for i in game_data:
            print("Score: {0}".format(i[0]))
    return traning_set



def build_model():
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))   
    model.add(Dense(52, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam())
    return model

def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 4)
    y = np.array([i[1] for i in training_set]).reshape(-1, 2)
    model.fit(X, y, epochs=10)


if __name__ == "__main__":
    N = 1000
    K = 50
    model = build_model()
    training_data = data_preparation(N, K, lambda s: random.randrange(0, 2))
    train_model(model, training_data)


    self_play_count = 10

    def predictor(s):
        return np.random.choice([0, 1], p=model.predict(s.reshape(-1, 4))[0])
    

    for i in range(self_play_count):
        K = (N//9 + K) // 2
        training_data = data_preparation(N, K, predictor)
        train_model(model, training_data)
    
    data_preparation(100, 100, predictor, True)
