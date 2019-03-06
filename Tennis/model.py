import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# critic takes actions and states of all agents

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class critic_network(nn.Module):
    def __init__(self, all_action_size, all_state_size, params):
        super(critic_network, self).__init__()
        
        self.seed = torch.manual_seed(4)
        
        self.all_state_size = all_state_size
        self.all_action_size = all_action_size
        self.V_max = params.V_max
        self.V_min = params.V_min
        self.delta = params.delta
        
        self.layer_1 = nn.Linear(self.all_state_size, 256)
        self.layer_2 = nn.Linear(256+self.all_action_size, 128)
        self.layer_3 = nn.Linear(128, params.atoms_number)
        
        self.bn1 = nn.BatchNorm1d(256)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(*hidden_init(self.layer_1))
        self.layer_2.weight.data.uniform_(*hidden_init(self.layer_2))
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)    
    
    def distr_to_q(self, distribution, device):
        sup = torch.arange(self.V_min, self.V_max + self.delta, self.delta).to(device)
        w = F.softmax(distribution, dim=1) * sup
        return w.sum(dim=1).unsqueeze(dim=-1)
    
    def forward(self, states, actions):
        if states.dim() == 1:
            states = torch.unsqueeze(states,0)
        x = F.relu(self.layer_1(states))
        x = self.bn1(x)
        x = torch.cat((x, actions), dim=1)
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

# actor takes value of own agent only
    
class actor_network(nn.Module):
    def __init__(self, action_size, state_size):
        super(actor_network, self).__init__()
        
        self.seed = torch.manual_seed(4)
        
        self.action_size = action_size
        self.state_size = state_size
        
        self.layer_1 = nn.Linear(self.state_size, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, self.action_size)
        
        self.bn1 = nn.BatchNorm1d(256)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(*hidden_init(self.layer_1))
        self.layer_2.weight.data.uniform_(*hidden_init(self.layer_2))
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)   
    
    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.layer_1(state))
        x = self.bn1(x)
        x = F.relu(self.layer_2(x))
        return F.tanh(self.layer_3(x))