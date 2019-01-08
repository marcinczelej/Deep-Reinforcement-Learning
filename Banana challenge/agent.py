import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from network_model import Network
from priority_replay import PrioritizedMemory
from collections import namedtuple, deque

MAX_BUFFER_SIZE = 5000
BATCH_SIZE = 64
ACTUALIZATION_INTERVAL = 5
TAU = 1e-3 
MIN_UPDATE = 1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, action_number, state_size, seed = 0, gamma = 0.99):
        
        self.action_number = action_number
        self.state_size = state_size
        self.targetNetwork = Network(self.state_size, self.action_number, seed).to(device)
        self.localNetwork = Network(self.state_size, self.action_number, seed).to(device)
        self.memoryBuffer = PrioritizedMemory(MAX_BUFFER_SIZE, BATCH_SIZE)
        self.current_step = 0
        self.gamma = gamma
        
        self.optimizer = optim.Adam(self.localNetwork.parameters(), lr = 0.001)

    def choose_action(self, state, eps):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.localNetwork.eval()
        with torch.no_grad():
            action_values = self.localNetwork(state)
        self.localNetwork.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_number))
        
    def step(self, state, action, reward, next_state, done):
        
        self.memoryBuffer.add(state, action, reward, next_state, done)
        
        self.current_step +=1
        
        if self.current_step%ACTUALIZATION_INTERVAL == 0 and len(self.memoryBuffer) >=BATCH_SIZE:
            buffer_data = self.memoryBuffer.get_batch()
            self.learn(buffer_data)

    def learn(self, buffer_data):
        """
        learning using:
            Experience Replay
            Double DQLearning
            dueling DQLearning
            delayed update
        """
        
        output_indexes, IS_weights, states, actions, rewards, next_states, dones = buffer_data
        
        # double Q learning
        best_predicted_action_number = self.localNetwork(next_states).detach().max(1)[1].unsqueeze(1)
        predicted_action_value = self.targetNetwork(next_states).detach().gather(1, best_predicted_action_number.view(-1, 1))
        # y_j calculation
        output_action_value = rewards + predicted_action_value*self.gamma*(1-dones)
        # expected values
        predicted_expected_action_value = self.localNetwork(states).gather(1, actions)

        # (y_j - expected value)**2
        
        #priority replay added last part *IS_WEIGHTS
        losses = F.mse_loss(predicted_expected_action_value, output_action_value, reduce=False)*IS_weights
        abs_error = losses+MIN_UPDATE
        self.memoryBuffer.update_batch(output_indexes, abs_error)
        
        self.optimizer.zero_grad()
        loss = losses.mean()
        loss.backward()
        self.optimizer.step()
        
        # updating target network
        for target_param, local_param in zip(self.targetNetwork.parameters(), self.localNetwork.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
            