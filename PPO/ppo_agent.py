import numpy as np
import torch
import torch.optimizer as optim

from model import Policy

class PPO_agent:
    def __init__(self, params):
        self.params = params
        self.net = Policy(4, 2) # state_size, action_size
        #self.net = Policy(params.state_size, params.action_size) # state_size, action_size
        
        if self.params.cuda:
            print("network is moved to cuda")
            self.net.cuda()
        
        self.optimizer = Adam(self.net.parameters(), lr = params.lr)
    
    # this method select action given input state
    # it return log_prob, value and action for given state
    # acording to current policy
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, log_prob, value = self.net.select_action(state)
        
        return action, log_prob, value
    
    # calculating surrogate function for PPO
    # - averaging advantage
    # - calcualting log_prob(s,a)
    # - calculating log_prob/old_log_prob
    # - clipping above ratio
    # - taking min value from clipped/no-clipped ratios (policy_loss)
    # - calculating losses : value_loss/entropy_loss
    # - backpropagating with given losses
    # - optimizing one step
    def evaluate_data(self, experience):
        # unpacking given experience data
        states, actions, rewards, dones, old_log_probs, values, gae_returns = experience
        
        # averaging advantage with baseline (value)
        advantages = gae_returns - values
        advantages = (advantages - advantages.mean())/(advantages.std() + 01e-5)
        
        # chaging all data into tensors of size (-1, 1)
        states = torch.FloatTensor(states).view(-1, 1).to(device)
        actions = torch.FloatTensor(actions).view(-1, 1).to(device)
        #old_log_probs = torch.FloatTensor(old_log_probs).view(-1, 1).to(device)
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device)
        gae_returns = torch.FloatTensor(gae_returns).view(-1, 1).to(device)
        values = torch.FloatTensor(values).view(-1, 1).to(device)
        
        #calculating new log_prob with given s,a
        new_log_probs, new_values, entropys = self.net.evaluate_inputs(states, actions)
        # calculating ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        ratio_without_clipping = ratio*advantage
        # clipping ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - self.params.clipping_value, 1.0 + self.params.clipping_value)*advantage
        # taking min value from both ratios ( policy_loss )
        policy_loss = torch.min(ratio_without_clipping, clipped_ratio).mean()
        
        # calculation losses (returns - values)^2
        value_loss = (gae_returns - new_values).pow(2).mean()
        # entropy loss = entropy*scaling_value
        entropy_loss = entropys*self.params.entropy_beta
        
        # backpropagation
        # zeroing gradient
        self.optimizer.zero_grad()
        # backpropagation
        (policy_loss + value_loss + entropy_loss).backward()
        # clipping gradinet ( TO DO )
        # optipmizer step to apply gradient
        self.optimizer.step()