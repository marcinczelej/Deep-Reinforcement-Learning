import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class a3cNetwork(nn.Module):
    def __init__(self, action_size):
        super(a3cNetwork, self).__init__()
        self.action_size = action_size
        
        #input #240 x 320 x 1
        
        self.layer_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)  # 118 x 158 x 128
        self.layer_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # 58 x 78 x 64
        self.layer_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) # 29 x 39 x 32
        
        self.batch_norm_32 = nn.BatchNorm2d(num_features=32)
        self.batch_norm_64 = nn.BatchNorm2d(num_features=64)
        
        self.lstm = nn.LSTMCell(self.calculate_flattered_size((1, 84, 84)), 128)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
       
        self.layer_4 = nn.Linear(in_features=128, out_features=action_size)
        self.critic_head = nn.Linear(128, 1)
        

    def calculate_flattered_size(self, input_size):
        input = Variable(torch.rand(1, *input_size))
        x = F.relu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = F.relu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.layer_3(x), 3, 2))
        return  x.data.view(1, -1).size(1)
    
    def forward(self, input):
        input, (hx, cx)= input
        x = F.relu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = self.batch_norm_32(x)
        x = F.relu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = self.batch_norm_32(x)
        x = F.relu(F.max_pool2d(self.layer_3(x), 3, 2))
        #x = self.batch_norm_64(x)
        
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.actor_forward(x), self.critic_head(hx), (hx, cx)

    def actor_forward(self, input_state):
        x = self.layer_4(input_state)
        prob = F.softmax(x, dim=1)
        log_prob = F.log_softmax(x, dim=1)
        entropy = (-prob*log_prob).sum(1)
        action = prob.multinomial(1)
        return {"prob" : prob,
               "log_prob" : log_prob,
               "entropy" : entropy,
               "action" : action}  
    