import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy

from Priority_replay import PrioritizedMemory
from Network import ActorNetwork, CriticNetwork
from OUNoise import OUNoise

class Agent():
    def __init__(self, action_size, state_size, params, device):
        self.batch_size = params.batch_size
        self.buffer_size = params.buffer_size
        self.tau = params.tau
        self.actor_lr = params.actor_lr
        self.critic_lr = params.critic_lr
        self.actor_weight_decay = params.actor_weight_decay
        self.critic_weight_decay = params.critic_weight_decay
        self.gamma = params.gamma
        self.params = params
        self.step_number =0
        self.device = device
        
        self.action_size= action_size
        self.state_size = state_size
        
        self.max_score = 40
        self.current_score = 0
        
        self.seed =  4
        
        self.actor_local = ActorNetwork(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = ActorNetwork(self.state_size, self.action_size, self.seed).to(device)
        
        self.critic_local = CriticNetwork(state_size, action_size, self.seed, params).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, self.seed, params).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr, weight_decay=self.critic_weight_decay)
        
        self.memory_buffer = PrioritizedMemory(self.buffer_size, self.batch_size, device)
        
        self.noise = OUNoise((20,self.action_size), self.seed)
        
    def select_action(self, state, device, noise = True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if noise:
            # the closer ihe score gets to the max score the less noise we add
            dampener = (self.max_score - np.min([self.max_score, self.current_score])) / self.max_score 
            action += self.noise.sample() * dampener
        action = np.clip(action,-1,1)
        return action
    
    def step(self, states, actions, rewards, next_states, dones):
        self.step_number +=1
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory_buffer.add(state, action, reward, next_state, done)
        if len(self.memory_buffer) >self.batch_size:
            batch = self.memory_buffer.get_batch()
            self.learn(batch)
    
    def learn(self, batch):
        
        #states, actions, rewards, next_states, dones = batch
        output_indexes, IS_weights, states, actions, rewards, next_states, dones = batch
        
        # critic update
        distribution = self.critic_local(states, actions)
        Q_value = self.actor_target(next_states)
        last_distribution = F.softmax(self.critic_target(next_states, Q_value), dim=1)
        projected_distribution = distr_projection(last_distribution, rewards.cpu().data.numpy(), dones.cpu().data.numpy(), self.params, gamma=(self.gamma**self.params.n_steps), device=self.device)
        prob_dist = -F.log_softmax(distribution, dim=1) * projected_distribution
        
        losses = prob_dist.sum(dim=1).view(-1,1)*IS_weights
        abs_error = losses+1e-5 
        
        critic_loss = losses.mean()
       
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # actor update
        self.actor_optimizer.zero_grad()
        
        action_choosen = self.actor_local(states)
        distribution = self.critic_local(states, action_choosen)
        actor_loss = -self.critic_local.distr_to_q(distribution, self.device).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        self.memory_buffer.update_batch(output_indexes, abs_error)
        
        if (self.step_number %100 == 0):
            #hard update
            self.soft_update_target(self.actor_local, self.actor_target, tau=1.)
            self.soft_update_target(self.critic_local, self.critic_target, tau=1.)
        else:
            # soft update
            self.soft_update_target(self.actor_local, self.actor_target, tau=self.tau)
            self.soft_update_target(self.critic_local, self.critic_target, tau=self.tau)    
        
    
    def soft_update_target(self, local_network, target_network, tau):
        for target, local in zip(target_network.parameters(), local_network.parameters()):
            target.data.copy_(tau*local.data + (1-tau)*target.data)
    
    def reset_noise(self):
        self.noise.reset()

def distr_projection(next_distr_v, rewards_v, dones_mask_t, params, gamma, device="cpu"):
    Vmin = params.V_min
    Vmax = params.V_max
    rewards = rewards_v.reshape(-1)
    dones_mask  = dones_mask_t.reshape(-1).astype(bool)
    next_distr = next_distr = next_distr_v.data.cpu().numpy()
    proj_distr = np.zeros((params.batch_size, params.atoms_number), dtype=np.float32)

    for atom in range(params.atoms_number):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * params.delta) * gamma))
        b_j = (tz_j - Vmin) / params.delta
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = (u == l).astype(bool)
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = (u != l).astype(bool)
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / params.delta
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)