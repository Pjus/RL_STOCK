# AI for temperature regulator for pump

# Importing the libraries

import numpy as np
import random # random samples from different batches (experience replay)
import os # For loading and saving brain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for using stochastic gradient descent
import torch.autograd as autograd # Conversion from tensor (advanced arrays) to avoid all that contains a gradient
# We want to put the tensor into a varaible taht will also contain a
# gradient and to this we need:
from torch.autograd import Variable
# to convert this tensor into a variable containing the tensor and the gradient

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1,keepdim=True).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size()) #?? list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros

# Creating the architecture of the Neural Network
class Network(nn.Module): #inherinting from nn.Module
    
    #Self - refers to the object that will be created from this class
    #     - self here to specify that we're referring to the object
    def __init__(self, input_size, nb_action): #[self,input neuroner, output neuroner]
        super(Network, self).__init__() #inorder to use modules in torch.nn
        # Input and output neurons
        self.lstm = nn.LSTMCell(input_size, 30) # making an LSTM (Long Short Term Memory) to learn the temporal properties of the input
        self.fcL = nn.Linear(30, nb_action) # full connection of the
        self.apply(weights_init) # initilizing the weights of the model with random weights
        self.fcL.weight.data = normalized_columns_initializer(self.fcL.weight.data, 0.01) # setting the standard deviation of the fcL tensor of weights to 0.01
        self.fcL.bias.data.fill_(0) # initializing the actor bias with zeros
        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms

		
		
    # For function that will activate neurons and perform forward propagation
    def forward(self, state):
        inputs, (hx, cx) = state # getting separately the input images to the tuple (hidden states, cell states)
        x = F.relu(self.lstm(inputs)) # forward propagating the signal from the input images to the 1st convolutional layer
        hx, cx = self.lstm(x, (hx, cx)) # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
        return self.fcL(x), (hx, cx) # returning the output of the actor (Q(S,A)), and the new hidden & cell states ((hx, cx))


# Implementing Experience Replay
# We know that RL is based on MDP
# So going from one state(s_t) to the next state(s_t+1)
# We gonna put 100 transition between state into what we call the memory
# So we can use the distribution of experience to make a decision
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #100 transitions
        self.memory = [] #memory to save transitions
    
    # pushing transitions into memory with append
    #event=transition
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity: #memory only contain 100 events
            del self.memory[0] #delete first transition from memory if there is more that 100
    
    # taking random sample
    def sample(self, batch_size):
        #Creating variable that will contain the samples of memory
        #zip =reshape function if list = ((1,2,3),(4,5,6)) zip(*list)= (1,4),(2,5),(3,6)
        #                (state,action,reward),(state,action,reward)  
        samples = zip(*random.sample(self.memory, batch_size))
        #This is to be able to differentiate with respect to a tensor
        #and this will then contain the tensor and gradient
        #so for state,action and reward we will store the seperately into some
        #bytes which each one will get a gradient
        #so that eventually we'll be able to differentiate each one of them
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma, lrate, T):
        self.gamma = gamma #self.gamma gets assigned to input argument
        self.T = T
        # Sliding window of the evolving mean of the last 100 events/transitions
        self.reward_window = []
        #Creating network with network class
        self.model = Network(input_size, nb_action)
        #creating memory with memory class
        #We gonna take 100000 samples into memory and then we will sample from this memory to 
        #to get a snakk number of random transitions
        self.memory = ReplayMemory(100000)
        #creating optimizer (stochastic gradient descent)
        self.optimizer = optim.Adam(self.model.parameters(), lr = lrate) #learning rate
        #input vector which is batch of input observations
        #by unsqeeze we create a fake dimension to this is
        #what the network expect for its inputs
        #have to be the first dimension of the last_state
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #Inilizing
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        #Q value depends on state
        #Temperature parameter T will be a positive number and the closer
        #it is to ze the less sure the NN will when taking an action
        #forexample
        #softmax((1,2,3))={0.04,0.11,0.85} ==> softmax((1,2,3)*3)={0,0.02,0.98} 
        #to deactivate brain then set T=0, thereby it is full random
        probs = F.softmax((self.model(Variable(state, volatile = True))*self.T),dim=1) # T=100
        #create a random draw from the probability distribution created from softmax
        action = probs.multinomial()
        return action.data[0,0]
    
    # See section 5.3 in AI handbook
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #next input for target see page 7 in attached AI handbook
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        #Using hubble loss inorder to obtain loss
        td_loss = F.smooth_l1_loss(outputs, target)
        #using  lass loss/error to perform stochastic gradient descent and update weights 
        self.optimizer.zero_grad() #reintialize the optimizer at each iteration of the loop
        #This line of code that backward propagates the error into the NN
        #td_loss.backward(retain_variables = True) #userwarning
        td_loss.backward(retain_graph = True)
		#And this line of code uses the optimizer to update the weights
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        #Updated one transition and we have dated the last element of the transition
        #which is the new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #After ending in a state its time to play a action
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")