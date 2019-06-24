import numpy
import torch

from ppo_agent import PPO_agent

class Learning_agent:
    def __init__(self, params, envs, agent):
        self.params = params
        self.envs = envs
        self.agent = agent
        
        # all needed tables of size (steps, environments number)
        # except of state. It should be (steps, [env_number, state_size])
        # state tensor
        self.states = torch.zeros((params.n_steps, params.n_envs, params.state_size), dtype=torch.float32, device = params.device)
        # float tensors
        self.gae_values = torch.zeros((params.n_steps, params.n_envs), dtype=torch.float32, device = params.device)
        self.log_probs = torch.zeros((params.n_steps, params.n_envs), dtype=torch.float32, device = params.device)
        self.values = torch.zeros((params.n_steps, params.n_envs), dtype=torch.float32, device = params.device)
        self.gae_returns = torch.zeros((params.n_steps, params.n_envs), dtype=torch.float32, device = params.device)
        # int tensors
        self.actions = torch.zeros((params.n_steps, params.n_envs), dtype=torch.int32, device = params.device)
        self.rewards = torch.zeros((params.n_steps, params.n_envs), dtype=torch.int32, device = params.device)
        self.dones = torch.zeros((params.n_steps, params.n_envs), dtype=torch.uint8, device = params.device)
    
    # collect old_prob trajectories 
    # for params.steps
    # during each step : action, value, log_prob is collected from given policy
    # next_state, reward, done from enviroments
    # all data are copied to proper tensors (using copy_)
    # last value is collected then for steps+1 time frame
    # after all steps gae is calculated and saved to proper tensor
    def collect_trajectories(self):
        
        state = self.envs.reset()
        
        # running for all steps
        for step in range(params.steps):
            # converting state to tensor
            state = torch.FloatTensor(states).to(self.params.device)
            action, log_prob, value = self.agent.select_action(state)
            
            # copy_(...) is used to keep gradient alive for original state, action, log_prob, value 
            # to keep them in coputational graph.
            # gradientw for self.states etc is 0 but for copy_(variable) ; variable that is copied is 1
            # so when backpropagation is running is will get good route form action to network etc.
            self.states[step].copy_(state)
            self.actions[ste].copy_(action)
            self.log_probs[step].copy_(log_prob)
            self.values[step].copy_(value)
            
            # getting state, areward, done
            state, reward, done, _ = self.envs.step(action.cpu())
            
            # same as above with copy_(...)
            # done, reward needs to be converted to tensors to push them into array
            # plus to enable gradient computation
            self.reward[step].copy_(reward)
            self.dones[step].copy_(done)
            
            # if done == True reset env
            if np.any(done):
                state = env.reset()
        
        # getting last value - needed for gae computation ( td_error part )
        _, _, last_value = self.agent.select_action(state)
            