# AI for Self Driving agent

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

#inherite from pytorch NN class that have functionalities to build Neural networks layer and more
class Network(nn.Module):
    #initialization of the network class and define the variables
    #input size is 5, 3 for agent sensors, and 2 for orientation and negatives orientation, those 5 variables represent the state.
    #The output is size of 3, witch is all possibleactions, either go straight, or right or left.
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #We create the layer of the network
        #we want one hidden layer, so we need 2 full connections
        #Linear create a full connection layer
        #The hidden layer is hyper params, and can be tested
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    #activate the neual in the neural networks and get the 
    #return the q_value the output the NN
    #also the final action can be choosen by the softmax function
    def forward(self, state):
        #the state is the same state 
        #activate the neural networks
        #X are the hideen neural networks
        x = F.relu(self.fc1(state))
        #return the q_values for all possible actions
        q_values = self.fc2(x)
        return q_values

    
# Implementing Experience Replay
# Looking at series of expetiences
# one time step is not usefull
# We put last 100 steps into memory long term meory
# We sample from memory , take ramdom batch from that memory
# to make next move
class ReplayMemory(object):
    
    def __init__(self, capacity):
        #the capacity is the size of the memory
        #the maximum number of transition to store in memory
        self.capacity = capacity
        #the memory is just a simple list
        self.memory = []
        
    #function tht update the memory and ensore the size of it
    #an even is a transition, it contains 4 elements
    #the last state, the new state, the last action, the last rewaard
    def push(self, event):
        self.memory.append(event)
        #the memory is orginised in for of a queue, where the first transition get delete the give space for a new transition
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    # the function take a randowm sample batch from the memory 
    def sample(self, batch_size):
        #list [[1, 2, 3], [4, 5, 6]]=zip=[[1, 4], [2, 5], [3, 6]]
        #we separete the batches of elemts into individual batches
        #one for old state other for new state ...
        samples = zip(*random.sample(self.memory, batch_size))
        #we convert these batches to list of torch variables
        #the torch store the variable and the gradient
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

#impliment the DQN model, 
class Dqn():
    #variable of the model, the gamma delay coeficient for the learning 
    def __init__(self, input_size, nb_action, gamma):
        #
        self.gamma = gamma
        #a slide window of the mean of the last 100 reward
        #we append the mean every time
        #evaluate the performence of the agent
        self.reward_window = []
        #instatia the neural network
        self.model = Network(input_size, nb_action)
        #instantial the memeory class
        self.memory = ReplayMemory(100000)
        #choose the optimizer of the nwtwork, we choose Adam Opimizer
        #We insert networks parameters, and the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #the transition event of transitions
        #state have 5 dimentions, 3 sign, 1-1orient
        #we add a dimention for the batch
        #we conservt the variables to bbatch of baribles
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #init the action last
        self.last_action = 0
        self.last_reward = 0
    
    #select the next action to play
    def select_action(self, state):
    
        
        #no grad to not use th egradient ot save memory
        with torch.no_grad():
            #feed the state to the neurle network to get the predictions of the actions
            #the output of the neural network is 3 q_values, we pass them to  softmax function to get
            #the final action to play
            #get the best action and still exploring the other actions
            #one porbability of the 3 action with the biggest prob of the heighest Q value
            #The exploration is parametirized by a teperature value to manipilat the exploration and exploitation 
            #modulate how NN sould be sure of witch action to decide to play
            #cloae to 0 less sure, hight more sure
            #T = 7, T = 100
            #the higher T the more the high q value is sure to be choosed
            #the certainty of witch the ation to play
            #generate a disrtibution of probabilities for all q values
            probs = F.softmax(self.model(Variable(state))*100, dim=1)
        #the F.softmax also take a random draw from the distribution for the final action
        action = probs.multinomial(1)
        #get the data from the variable
        return action.data[0,0]
    
    #train the deep neural network, the process of forpropagation and backpropagation using stochastic gradient desecnt
    #pass the transition in batches
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #we feed the state into the model, and we get the q avle for all action
        #we specify the to choose th q value for the action that was actually choosen in that case
        #using the action batch
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #get the prediction for the next state from the model also, and we want this time the maximin q value for the each state in the batch with repect to actions, 
        #get the maximum value of next state acording to according to all actions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #we calculate the target value fron the next state and reward and gamma
        target = self.gamma*next_outputs + batch_reward
        #We calculate the temporal difference
        td_loss = F.smooth_l1_loss(outputs, target)
        #train the model with the optimizer
        self.optimizer.zero_grad()
        #do the backpropagation
        td_loss.backward(retain_graph  = True)
        #update the weight of the models
        self.optimizer.step()
    
    #update the action, state, reward
    #connection the agent with the environment
    #we also select the next action
    def update(self, reward, new_signal):
        #transform the input state into torch tensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #insert the ternsition into the memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        
        #select a new action to play in the environment
        action = self.select_action(new_state)
        
        #We accumulate the experiences and learn if we have enough expericences
        if len(self.memory.memory) > 100:
            #We use 100 experiences to learn , 100 is the batch size now
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            #We use learn function to do training
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #update the variable
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        #Update the reward window, by calculating the mean of the last 1000 rewards
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        #return the selected action
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