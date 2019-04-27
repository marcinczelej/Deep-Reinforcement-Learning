import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class a3cNetwork(nn.Module):
    def __init__(self, action_size):
        super(a3cNetwork, self).__init__()
        self.action_size = action_size
        
        #input #240 x 320 x 1
        
        self.layer_1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5, stride=2, padding=1)  # 118 x 158 x 128
        self.layer_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=2, padding=1) # 58 x 78 x 64
        self.layer_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=2, padding=1) # 29 x 39 x 32
        self.layer_4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride=2, padding=1) # 29 x 39 x 32
        
        self.batch_norm_64 = nn.BatchNorm2d(num_features=64)
        
        self.lstm = nn.LSTMCell(self.calculate_flattered_size((1, 100, 120)), 256)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
       
        self.layer_5 = nn.Linear(in_features=256, out_features=64)
        self.layer_6 = nn.Linear(in_features=64, out_features=action_size)
        self.critic_head = nn.Linear(64, 1)
        
        self.train()

    def calculate_flattered_size(self, input_size):
        input = Variable(torch.rand(1, *input_size))
        x = F.elu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = F.elu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = F.elu(F.max_pool2d(self.layer_3(x), 3, 2))
        x = F.elu(F.max_pool2d(self.layer_4(x), 3, 2))
        return  x.data.view(1, -1).size(1)
    
    def forward(self, input):
        input, (hx, cx)= input
        x = F.elu(F.max_pool2d(self.layer_1(input), 3, 1))
        x = self.batch_norm_64(x)
        x = F.elu(F.max_pool2d(self.layer_2(x), 3, 2))
        x = self.batch_norm_64(x)
        x = F.elu(F.max_pool2d(self.layer_3(x), 3, 2))
        x = self.batch_norm_64(x)
        x = F.elu(F.max_pool2d(self.layer_4(x), 3, 2))
        
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        x = self.layer_5(x)
        return self.actor_forward(x), self.critic_head(x), (hx, cx)

    def actor_forward(self, input_state):
        x = self.layer_6(input_state)
        prob = F.softmax(x, dim=1)
        log_prob = F.log_softmax(x, dim=1)
        entropy = (-prob*log_prob).sum(1)
        action = prob.multinomial(1)
        return {"prob" : prob,
               "log_prob" : log_prob,
               "entropy" : entropy,
               "action" : action}  
    