import gym
from model import a3cNetwork
import torch

def train(process_number, params, shared_model, shared_optimizer, device):
    env = gym.make(params.env_name)
    env.seed(params.seed+process_number)
    torch.manual_seed(params.seed+process_number)
    state = env.reset()
    
    model = a3cNetwork(params.state_space, params.action_space, device).to(device)
    
    while True:
        model.load_state_dict(shared_model.state_dict())
        values = []     # TD part, advantage part
        rewards = []    # to collect rewards ( reinforce + TD part)
        entropys = []   # to make use of entropy
        log_probs = []  # needed for reinforce part
        
        for i in range(params.n_steps):
            actor_return, value = model(state)
            values.append(value)
            entropys.append(actor_return["entropy"]) 
            action = actor_return["action"].data.numpy()[0][0]
            log_probs.append(actor_return["log_prob"].gather(1, actor_return["action"]))
            state, reward, done,_ = env.step(action)
            rewards.append(reward)
            if done:
                state = env.reset()
                break
        R = torch.zeros(1,1)
        if not done:
            _, value = model(state)
            R = value.data
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1,1)
        
        # GAE A = Q - V
        for i in reversed(range(len(rewards))):
            #value loss
            R = rewards[i] + R*params.gamma
            advantage_i = R - values[i]
            value_loss = value_loss + 0.5*advantage_i.pow(2)
            #policy loss
            TD = rewards[i] + params.gamma*values[i+1] - values[i]
            gae = gae*params.gamma*params.tau + TD
            policy_loss = policy_loss - log_probs[i]*gae - 0.01*entropys[i]
        loss = policy_loss + 0.5*value_loss
        shared_optimizer.zero_grad()
        loss.backward()
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                break
            shared_param._grad = param.grad
        shared_optimizer.step()
