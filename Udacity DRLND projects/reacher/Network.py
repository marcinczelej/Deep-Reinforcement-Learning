import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_units = [512, 256]):
        super(ActorNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        
        self.linear_1 = nn.Linear(self.state_size, hidden_units[0])
        self.linear_2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.linear_3 = nn.Linear(hidden_units[1], self.action_size)
        
        self.batch_norm = nn.BatchNorm1d(hidden_units[0])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.linear_1.weight.data.uniform_(*hidden_init(self.linear_1))
        self.linear_2.weight.data.uniform_(*hidden_init(self.linear_2))
        self.linear_3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = self.linear_1(state)
        x = F.relu(x)
        x = F.relu(self.linear_2(x))
        return F.tanh(self.linear_3(x))

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, params, hidden_units = [512, 256]):
        super(CriticNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size
        self.V_max = params.V_max
        self.V_min = params.V_min
        self.delta = params.delta

        self.linear_1 = nn.Linear(self.state_size, hidden_units[0])
        self.linear_2 = nn.Linear(hidden_units[0] + action_size, hidden_units[1])
        self.linear_3 = nn.Linear(hidden_units[1], params.atoms_number)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear_1.weight.data.uniform_(*hidden_init(self.linear_1))
        self.linear_2.weight.data.uniform_(*hidden_init(self.linear_2))
        self.linear_3.weight.data.uniform_(-3e-3, 3e-3)
    
    def distr_to_q(self, distribution, device):
        sup = torch.arange(self.V_min, self.V_max + self.delta, self.delta).to(device)
        w = F.softmax(distribution, dim=1) * sup
        return w.sum(dim=1).unsqueeze(dim=-1)
    
    def forward(self, state, action):
        x = F.relu(self.linear_1(state))
        x = torch.cat((x,action), dim=1)
        x = F.relu(self.linear_2(x))
        
        return self.linear_3(x)