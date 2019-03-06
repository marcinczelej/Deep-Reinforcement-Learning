import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from OUNoise import OUNoise
from Priority_replay import PrioritizedMemory
from model import critic_network, actor_network

class DDPGAgent():
    def __init__(self, state_size, action_size, device, agents_amount, agent_number, params):
        self.state_size = state_size
        self.action_size = action_size,
        self.device = device
        self.agent_number = agent_number-1
        self.step_number = 0
        self.params = params
        self.agents_amount = agents_amount
        self.eps = params.EPS_START
        self.eps_decay = 1/(params.EPS_EP_END*params.updates_per_step)
        
        self.critic_local = critic_network(action_size*agents_amount, state_size*agents_amount, params).to(self.device)
        self.critic_target = critic_network(action_size*agents_amount, state_size*agents_amount, params).to(self.device)
        
        self.actor_local = actor_network(action_size, state_size).to(self.device)
        self.actor_target = actor_network(action_size, state_size).to(self.device)
        
        self.critic_optimizer=optim.Adam(self.critic_local.parameters(), lr=self.params.critic_lr, weight_decay=params.critic_weight_decay)
        self.actor_optimizer=optim.Adam(self.actor_local.parameters(), lr=self.params.actor_lr, weight_decay=self.params.actor_weight_decay)
        
        self.noise = OUNoise(self.action_size, 4)
        
        self.memory_buffer = PrioritizedMemory(self.params.buffer_size, self.params.batch_size, self.device)
    
    def reset_noise(self):
        self.noise.reset()
    
    def select_action(self, input_state, noiseOn = True):
        input_state = torch.from_numpy(input_state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(input_state).cpu().data.numpy()
        self.actor_local.train()
        if noiseOn == True:
            actions += self.eps * self.noise.sample()
        return np.clip(actions, -1, 1)
    
    # state and action hav all agents informations [agent_number, state/action]
    def step(self, state, action, reward, next_state, done):
        self.memory_buffer.add(state, action, reward, next_state, done)
        self.step_number += 1

    def learn(self, batch, target_next, local_next):
        
        output_indexes, IS_weights, states, actions, rewards, next_states, dones = batch

        # converted from batch_szie, agents, size --> batch_size, size*agent_size
        critic_states = states.view(self.params.batch_size, -1)
        critic_next_states = next_states.view(self.params.batch_size, -1)
        critic_actions = actions.view(self.params.batch_size, -1)

        # critic update
        distribution = self.critic_local(critic_states, critic_actions)

        last_distribution = F.softmax(self.critic_target(critic_next_states, target_next), dim=1)
        projected_distribution = distr_projection(last_distribution, rewards[:,self.agent_number].cpu().data.numpy(), dones[:,self.agent_number].cpu().data.numpy(), gamma=(self.params.gamma**self.params.n_steps), params=self.params, device=self.device)
        prob_dist = -F.log_softmax(distribution, dim=1) * projected_distribution
        
        losses = prob_dist.sum(dim=1).view(-1,1)*IS_weights
        abs_error = losses+1e-5 
        
        critic_loss = losses.mean()
       
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # actor update
        self.actor_optimizer.zero_grad()

        distribution = self.critic_local(critic_states, local_next)
        actor_loss = -self.critic_local.distr_to_q(distribution, self.device).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if (self.step_number %1000 == 0):
            #hard update
            self.soft_update_target(self.actor_local, self.actor_target, tau=1.)
            self.soft_update_target(self.critic_local, self.critic_target, tau=1.)
        else:
            # soft update
            self.soft_update_target(self.actor_local, self.actor_target, tau=self.params.tau)
            self.soft_update_target(self.critic_local, self.critic_target, tau=self.params.tau)    
        
        self.eps -= self.eps_decay
        self.eps = max(self.eps, self.params.EPS_FINAL)
        self.noise.reset()
        
        return output_indexes, abs_error
    
    def soft_update_target(self, local_network, target_network, tau):
        for target, local in zip(target_network.parameters(), local_network.parameters()):
            target.data.copy_(tau*local.data + (1-tau)*target.data)
    
    def reset_noise(self):
        self.noise.reset()

def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, params, device="cpu"):

    Vmin = params.V_min
    Vmax = params.V_max
    rewards = rewards_v.reshape(-1)
    dones_mask  = dones_mask_t.reshape(-1).astype(bool)
    next_distr = next_distr_v.data.cpu().numpy()
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
    