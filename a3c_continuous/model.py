import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(Network, self).__init__()
        
        self.device = device
        
        self.linear_1 = nn.Linear(state_size, 256)
        self.linear_2 = nn.Linear(256, 256)
        self.linear_3 = nn.Linear(256, 128)
        self.linear_4 = nn.Linear(128, 128)
        
        self.actor_head = nn.Linear(128, action_size.shape[0])
        self.actor_head2 = nn.Linear(128, action_size.shape[0])
        self.critic_head = nn.Linear(128, 1)
        
        self.lstm = nn.LSTMCell(128, 128)
        
        self.linear_1.weight.data.mul_(nn.init.calculate_gain('leaky_relu'))
        self.linear_2.weight.data.mul_(nn.init.calculate_gain('leaky_relu'))
        self.linear_3.weight.data.mul_(nn.init.calculate_gain('leaky_relu'))
        self.linear_4.weight.data.mul_(nn.init.calculate_gain('leaky_relu'))
        
        torch.nn.init.xavier_uniform_(self.actor_head.weight, gain=nn.init.calculate_gain('leaky_relu'))
        torch.nn.init.xavier_uniform_(self.actor_head2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        torch.nn.init.xavier_uniform_(self.critic_head.weight, gain=nn.init.calculate_gain('leaky_relu'))
        
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        self.actor_head.bias.data.fill_(0)
        self.actor_head2.bias.data.fill_(0)
        self.critic_head.bias.data.fill_(0)
        
        self.train()
        
    def forward(self, input_state):
        x, (hx, cx) = input_state
        x = x.unsqueeze(0)
        
        x = F.leaky_relu(self.linear_1(x), 0.1)
        x = F.leaky_relu(self.linear_2(x), 0.1)
        x = F.leaky_relu(self.linear_3(x), 0.1)
        x = F.leaky_relu(self.linear_4(x), 0.1)
        
        hx, cx = self.lstm(x, (hx,cx))
        x = hx

        return self.critic_head(x), F.softsign(self.actor_head(x)), self.actor_head2(x), (hx, cx)
