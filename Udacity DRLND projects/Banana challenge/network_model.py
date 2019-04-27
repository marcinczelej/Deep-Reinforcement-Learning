import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):

    def __init__(self, state_size, action_size, seed = 1400, hidden_units = [128, 64]):

        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layer1 = nn.Linear(state_size, hidden_units[0])
        self.layer2 = nn.Linear(hidden_units[0], hidden_units[1])
        
        # normal initializtion
        nn.init.xavier_uniform_(self.layer1.weight.data)
        nn.init.xavier_uniform_(self.layer2.weight.data)
        self.layer1.bias.data.fill_(0)
        self.layer2.bias.data.fill_(0)
        
        # Dueling part
        self.advantageValues = nn.Linear(hidden_units[1], action_size)
        self.stateValues = nn.Linear(hidden_units[1], 1)
     
        self.advantageValues.bias.data.fill_(0)
        self.stateValues.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.advantageValues.weight.data)
        nn.init.xavier_uniform_(self.stateValues.weight.data)
        # end of dueling part

    def forward(self, state):
        
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        advantages_layer = self.advantageValues(x)
        value_layer = self.stateValues(x)

        # result for dueling DDQN
        return advantages_layer + value_layer - advantages_layer.mean()