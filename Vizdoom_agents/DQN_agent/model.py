import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Base_network(nn.Module):
    def __init__(self, action_size, is_dueling = False):
        super(Base_network, self).__init__()
        self.action_size = action_size
        self.is_dueling = is_dueling
        
        #input #240 x 320 x 1
        
        self.layer_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)  # 118 x 158 x 128
        self.layer_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # 58 x 78 x 64
        self.layer_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) # 29 x 39 x 32
        
        self.lstm = nn.LSTMCell(self.calculate_flattered_size((1, 84, 84)), 40)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
       
        self.layer_4 = nn.Linear(in_features=40, out_features=40)
        
        if is_dueling==True:
            self.advantage_layer = nn.Linear(in_features = 40, out_features = self.action_size)
            self.value_layer = nn.Linear(in_features = 40, out_features = 1)

    def calculate_flattered_size(self, input_size):
        input = Variable(torch.rand(1, *input_size))
        x = F.relu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = F.relu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.layer_3(x), 3, 2))
        return  x.data.view(1, -1).size(1)
    
    def base_forward(self, input):
        input, (hx, cx)= input
        x = F.relu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = F.relu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.layer_3(x), 3, 2))
        
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
        return x, (hx, cx)
    
    def head_forward(self, input):
        x = F.relu(self.layer_4(input))
        if self.is_dueling==True:
            advantage = self.advantage_layer(x)
            value = self.value_layer(x)
            return advantage + value - advantage.mean()
        else:
            return x

# inherit Base_network, without dueling        
        
class Network(Base_network):
    def __init__(self, action_size):
        super(Network, self).__init__(action_size, False)
        self.action_size = action_size
        
        self.layer_5 = nn.Linear(in_features=40, out_features=self.action_size)
    
    def forward(self, input):
        x, (hx, cx) = self.base_forward(input)
        x = self.head_forward(x)
        x = self.layer_5(x)
        return x, (hx, cx)
    
# inherit Base_network, with dueling   
class Dueling_network(Base_network):
    def __init__(self, action_size):
        super(Dueling_network, self).__init__(action_size, True)
        self.action_size = action_size

    def forward(self, input):
        x, (hx, cx) = self.base_forward(input)
        x = self.head_forward(x)
        return x, (hx, cx)
        