import numpy as np
import torch
from collections import namedtuple

MIN_UPDATE = 1e-5

class SumTree(object):
    current_size = 0
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.data_type = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.data = np.zeros(max_size, dtype=self.data_type)
        
        self.priorities = np.zeros(2 * max_size - 1)
        
    def add(self, priority, state, action, reward, next_state, done):
        
        tree_node = self.max_size + self.current_size -1
        data = self.data_type(state, action, reward, next_state, done)
        self.data[self.current_size] = data
        
        self.update(priority, tree_node)
        
        self.current_size +=1
        
        if self.current_size >= self.max_size:
            self.current_size = 0
        
    def update(self, priority, current_position):
        
        priority_difference = priority - self.priorities[current_position]
        
        self.priorities[current_position] = priority
        
        while current_position !=0:
            current_position = (current_position-1)//2
            self.priorities[current_position] += priority_difference            
        
    def get(self, priority):
        
        parent_index=0
        
        while True:
            left_index = 2*parent_index +1
            right_index = left_index +1
            
            if left_index >= len(self.priorities):
                leaf_index = parent_index
                break
            
            if priority <= self.priorities[left_index]:
                parent_index = left_index
            else:
                priority -= self.priorities[left_index]
                parent_index = right_index
        data_index = leaf_index - self.max_size +1
        
        return leaf_index, self.priorities[leaf_index], self.data[data_index]
    
    def get_element_amount(self):
        return self.current_size
    
    def get_max_priority(self):
        return np.max(self.priorities[-self.max_size:])
    
    def get_total_priority(self):
        return self.priorities[0]

class PrioritizedMemory(object):
    
    IS_beta = 0.4
    IS_beta_change = 0.001
    absolute_error_upper = 0.001
    PER_a = 0.6
    
    def __init__(self, max_size, batch_size, device):
        self.tree = SumTree(max_size)
        self.batch_size = batch_size
        self.device = device
        
    def add(self, state, action, reward, next_state, done):
        
        priority = self.tree.get_max_priority()
        
        if priority == 0:
            priority = self.absolute_error_upper

        self.tree.add(priority, state, action, reward, next_state, done)
        
    def get_batch(self):
        
        IS_weights = np.zeros((self.batch_size, 1))  # impoertance sampling weights
        output_index = np.zeros((self.batch_size, 1))
        experiences = []
        
        prio_segment = self.tree.get_total_priority()/self.batch_size
        self.IS_beta = np.min([1.0, self.IS_beta + self.IS_beta_change])
        
        # w(i) normalized
        min_prop = np.min(self.tree.get_max_priority())/self.tree.get_total_priority()
        IS_max = (min_prop * self.batch_size) ** (-self.IS_beta)
        
        for i in range(self.batch_size):
            
            a, b = prio_segment*i, prio_segment*(i+1)
            priority = np.random.uniform(a,b)

            index, priority, data = self.tree.get(priority)
            # P(j) = p(j)**PER_a/sum props
            sampling_propability = priority/self.tree.get_total_priority()
            
            #w(i) = (N*P(j))**(-IS_beta)
            IS_weights[i,0] = np.power(self.batch_size*sampling_propability, -self.IS_beta)/(IS_max)
            output_index[i,0] = index
            experiences.append(data)
        
        states =  torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(np.vstack(IS_weights)).float().to(self.device)
        indexes = torch.from_numpy(np.vstack(output_index)).long().to(self.device)
        
        return indexes, weights, states, actions, rewards, next_states, dones
    
    def update_batch(self, indexes, TD_error):
        TD_error += 0.01
        clipped_errors = np.minimum(TD_error.detach().cpu().numpy(), self.absolute_error_upper)
        
        ps = np.power(clipped_errors, self.PER_a)
        
        for priority, position in zip(ps, indexes.detach()):
            self.tree.update(priority, position.data[0].item())
            
    def __len__(self):
        return self.tree.get_element_amount()