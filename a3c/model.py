import torch
import torch.nn as nn
import torch.nn.functional as F

class a3cNetwork(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(a3cNetwork, self).__init__()
        self.device = device
        
        self.layer_1 = nn.Linear(state_size, 16)
        
        self.actor_head = nn.Linear(16, action_size)
        self.critic_head = nn.Linear(16, 1)
        
    def forward(self, input_state):
        input_state = torch.from_numpy(input_state).float().unsqueeze(0).to(self.device)
        
        x = F.relu(self.layer_1(input_state))
        
        return self.actorHead(x), self.critic_head(x)
    
    def actorHead(self, input_state):
        x = self.actor_head(input_state)
        prob = F.softmax(x, dim=1)
        log_prob = F.log_softmax(x, dim=1)
        entropy = (-prob*log_prob).sum(1)
        action = prob.multinomial(1)
        return {"prob" : prob,
               "log_prob" : log_prob,
               "entropy" : entropy,
               "action" : action}