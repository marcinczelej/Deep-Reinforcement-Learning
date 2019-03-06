import numpy as np
import torch

from DDPG_agent import DDPGAgent
from Priority_replay import PrioritizedMemory

class MADDPGWrapper():
    def __init__(self, action_size, state_size, device, params, agents_amount):
        self.params = params
        self.agents_amount = agents_amount

        self.memory_buffer = PrioritizedMemory(params.buffer_size, params.batch_size, device)
        self.agents = [DDPGAgent(state_size, action_size, device, agents_amount, 1, params),
        DDPGAgent(state_size, action_size, device, agents_amount, 2, params)]

    def prepopulateBuffer(self, brain_name, env):
        print("prepopulation start")
        counter =0
        while True:
            if counter >= self.params.buffer_size-1:
                break
            env_info = env.reset(train_mode=True)[brain_name]
            next_states = states = env_info.vector_observations
            while counter <= self.params.buffer_size-1:
                n_step_rewards = np.zeros(self.agents_amount)
                if counter%1000 ==0:
                    print(counter)
                for i in range(self.params.n_steps-1):
                    states = next_states
                    actions = self.select_actions(states)
                    env_info = env.step(actions)[brain_name]
                    next_states = env_info.vector_observations
                    rewards = env_info.rewards
                    dones = env_info.local_done
                    n_step_rewards += [reward * (self.params.gamma**i) for reward in rewards]
                if np.any(dones):
                    break
                self.memory_buffer.add(states, actions, n_step_rewards, next_states, dones)
                states = next_states
                counter+=1
        print("prepopulation end")

    def select_actions(self, states, noise_enabled = True):
        actions = [agent.select_action(state, noise_enabled) for agent, state in zip(self.agents, states)]
        return actions

    def step(self, states, actions, n_step_rewards, next_states, dones):
        self.memory_buffer.add(states, actions, n_step_rewards, next_states, dones)
        for agent in self.agents:
            agent.step(states, actions, n_step_rewards, next_states, dones)
            if len(self.memory_buffer) >= self.params.batch_size:
                for i in range(self.params.updates_per_step):
                    batch = self.memory_buffer.get_batch()
                    output_indexes, IS_weights, states, actions, rewards, next_states, dones = batch
                    target_next = self.get_action_target(next_states)
                    local_next = self.get_action_local(states)
                    (output_indexes, abs_error) = agent.learn(batch, target_next, local_next)
                    self.memory_buffer.update_batch(output_indexes, abs_error)

    def get_action_target(self, states):
        agents_states = []
        for i in range(self.agents_amount):
            agents_states.append(states[:,i,:])
        Q_values = torch.cat([agent.actor_target(state) for agent, state in zip(self.agents, agents_states)], dim=1)
        return Q_values
    
    def get_action_local(self, states):
        agents_states = []
        for i in range(self.agents_amount):
            agents_states.append(states[:,i,:])
        Q_values = torch.cat([agent.actor_local(state) for agent, state in zip(self.agents, agents_states)], dim=1)
        return Q_values
    
    def save_model(self, name):
        torch.save(self.agents[0].actor_local.state_dict(), 'checkpoint_first_actor_' + name)      
        torch.save(self.agents[0].critic_local.state_dict(), 'checkpoint_first_critic_' + name)
        torch.save(self.agents[1].actor_local.state_dict(), 'checkpoint_second_actor_' + name)      
        torch.save(self.agents[1].critic_local.state_dict(), 'checkpoint_second_critic_' + name)
    
    def load_model(self):
        self.agents[0].actor_local.load_state_dict(torch.load('checkpoint_first_actor_final.pth'))
        self.agents[0].critic_local.load_state_dict(torch.load('checkpoint_first_critic_final.pth'))
        self.agents[1].actor_local.load_state_dict(torch.load('checkpoint_second_actor_final.pth'))
        self.agents[1].critic_local.load_state_dict(torch.load('checkpoint_second_critic_final.pth'))
        