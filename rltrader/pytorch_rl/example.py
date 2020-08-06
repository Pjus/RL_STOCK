# import gym
# import math
# import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple
# from itertools import count
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

# env = gym.make('CartPole-v0').unwrapped

# # # matplotlib 설정
# # is_ipython = 'inline' in matplotlib.backend()
# # if is_ipython:
# #     from IPython import display

# plt.ion()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """transition 저장"""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
    
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
    
#     def __len__(self):
#         return len(self.memory)

# # Networks
# class DQN(nn.Module):

#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         # (3, 16, 5) input channel size, output volume size, kernel size(filter size)
#         # 
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 - nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)

#         # Linear 입력의 연결 숫자는 conv2d 계층의 출력과 입력 이미지의 크기에 따라 결정되기 때문에 따로 계산을 해야합니다
#         def conv2d_size_out(size, kernel_size=5, stride=2):
#             return (size - (kernel_size - 1) - 1) // stride + 1
        
#         convw = conv2d_size_out(size, kernel_size=5, stride=2)





    
from collections import deque

obs_open = deque(np.zeros(20), maxlen=20)
obs_open.append(20)
print(obs_open)