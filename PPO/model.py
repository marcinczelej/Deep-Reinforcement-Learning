import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, state_size, action_size, layer_size = [32, 16]):
        super(Policy, self).__init__()
        
        self.layer_1 = nn.Linear(state_size, layer_size[0])
        
        self.hidden = nn.ModuleList()
        for i in range(len(layer_size)-1):
            print("creating layer ", layer_size[i], " , ", layer_size[i+1])
            self.hidden.append(nn.Linear(layer_size[i], layer_size[i+1]))
        
        self.value_layer = nn.Linear(layer_size[-1], 1)
        self.actor_layer = nn.Linear(layer_size[-1], action_size)
        
        self.train()

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return x
    
    # returning log_prob, entropy
    def calculate_distributions(self, state, action):
        x = self.actor_layer(state)
        #calcualting log_prob and prob
        probs = F.softmax(x)
        log_probs = F.log_softmax(x)
        
        #calculating entropy
        entropy = -(log_probs * probs).sum(-1).mean()
        return log_probs.gather(1, action), entropy
    
    # returning action based on given state
    def choose_action(self, state):
        x = self.actor_layer(state)
        prob = F.softmax(x)
        return prob.multinomial(1)
    # -------------------------------------------------- METHODS TO BE USED ---------------------------
    # returning action, log_prob, value
    def select_action(self, input_state):
        print("select_action")
        # calling forward of Policy
        x = self(input_state)
        # calculating value
        value = self.value_layer(x)
        # choosing action
        action = self.choose_action(x)
        # claculating log_prob, entropy
        log_prob, _ =  self.calculate_distributions(x, action)
        return action, log_prob, value
                            
    # returning log_prob, value, entropy
    def evaluate_inputs(self, input_state, action):
        print("evaluate_inputs")
        x = self(input_state)
        value = self.value_layer(x)
        log_prob, entropy = self.calculate_distributions(x, action)
        return log_prob, value, entropy