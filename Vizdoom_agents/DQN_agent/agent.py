import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from model import Network, Dueling_network
from memory_replay import PrioritizedMemory

class params():
    batch_size = 64
    buffer_size = 10000
    learning_rate = 0.0001
    gamma = 0.99
    learnig_interval = 5
    min_update = 1e-5
    tau = 0.01
    n_steps = 2

class DQN_agent():
    def __init__(self, action_size, device, writer, is_dueling=False):
        self.action_size = action_size
        self.device = device
        self.memoryBuffer = PrioritizedMemory(params.buffer_size, params.batch_size, device)
        self.current_step = 0
        self.writer = writer

        if is_dueling:
            print("Creating dueling network")
            self.local_network = Dueling_network(action_size).to(device)
            self.target_network = Dueling_network(action_size).to(device)
            self.predict_network = Dueling_network(action_size).to(device)
        else:
            print("Creating network without dueling")
            self.local_network = Network(action_size).to(device)
            self.target_network = Network(action_size).to(device)
            self.predict_network = Network(action_size).to(device)
        
        for predict_param, local_param in zip(self.predict_network.parameters(), self.local_network.parameters()):
            predict_param.data.copy_(local_param.data)
        
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=params.learning_rate)
    
    def select_action(self, input_state, epsilon, LSTM_input):
        input_state = torch.from_numpy(input_state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        (hx, cx) = LSTM_input
        self.predict_network.eval()
        with torch.no_grad():
            actions, _ = self.predict_network((input_state, (hx, cx)))
        self.predict_network.train()
        
        # epsilon greedy policy
        if random.random() > epsilon:
            return np.argmax(actions.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def step(self, episode, state, action, reward, next_state, done, hx, cx):
        self.memoryBuffer.add(state, action, reward, next_state, done)
        
        lstm_values = (hx, cx)
        self.current_step +=1
        if self.current_step%params.learnig_interval == 0:
            for i in range(3):
                experience = self.memoryBuffer.get_batch()
                lstm_values = self.learn(episode, experience, hx, cx)
        return lstm_values
            
    def learn(self, episode, experience, hx, cx):
        output_indexes, IS_weights, states, actions, rewards, next_states, dones = experience
        
        Q_argmax, _ = self.local_network((next_states.unsqueeze(1), (hx, cx)))
        argmax = Q_argmax.detach().max(1)[1].unsqueeze(1)
        Q_next, _ = self.target_network((next_states.unsqueeze(1), (hx, cx)))
        Q_next = Q_next.detach().gather(1, argmax.view(-1, 1))
        
        y_i = rewards + (params.gamma**params.n_steps)*(1-dones)*Q_next
        Q_target, (hx, cx) = self.local_network((states.unsqueeze(1), (hx, cx)))
        Q_target = Q_target.gather(1, actions)
        
        losses = F.mse_loss(Q_target, y_i, reduce=False)*IS_weights
        abs_error = losses+params.min_update
        self.memoryBuffer.update_batch(output_indexes, abs_error)
        
        self.optimizer.zero_grad()
        loss = losses.mean()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Reward Loss/Reward', loss, episode)
        
        for target_param, local_param, predict_param in zip(self.target_network.parameters(), self.local_network.parameters(), self.predict_network.parameters()):
            target_param.data.copy_(params.tau*local_param.data + (1.0-params.tau)*target_param.data)
            predict_param.data.copy_(local_param.data)  
        return (hx, cx)
