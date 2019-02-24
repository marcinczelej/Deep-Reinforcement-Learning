import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.action_size = action_size
        
        #input #240 x 320 x 1
        
        self.layer_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)  # 118 x 158 x 128
        self.layer_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # 58 x 78 x 64
        self.layer_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) # 29 x 39 x 32
        
        self.lstm = nn.LSTMCell(self.calculate_flattered_size((1, 84, 84)), 256)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        self.layer_4 = nn.Linear(in_features=256, out_features=64)
        self.layer_5 = nn.Linear(in_features=64, out_features=self.action_size)
        
    def calculate_flattered_size(self, input_size):
        input = Variable(torch.rand(1, *input_size))
        x = F.relu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = F.relu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.layer_3(x), 3, 2))
        return  x.data.view(1, -1).size(1)
    
    def forward(self, input):
        input, (hx, cx)= input
        x = F.relu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = F.relu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.layer_3(x), 3, 2))
        
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
        x = F.relu(self.layer_4(x))
        x = self.layer_5(x)
        return x, (hx, cx)