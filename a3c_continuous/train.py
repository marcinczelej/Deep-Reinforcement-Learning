import gym
import math
import torch

import numpy as np

from torch.autograd import Variable
from model import Network

def gaussianDistribution(x, mu, sigma, pi):
    scaling_factor = 1./(2*pi.expand_as(sigma)*sigma).sqrt()
    exp_part = (-1*(x - mu).pow(2)/(2*sigma)).exp()
    return scaling_factor*exp_part

def CalculateDistribution(value, mu, sigma):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi)
    mu = torch.clamp(mu, -1.0, 1.0)
    sigma = torch.nn.functional.softplus(sigma) + 1e-5
    epsilon = torch.randn(mu.size())
    epsilon = Variable(epsilon)
    action = (mu + sigma.sqrt()*epsilon).data
    prob = gaussianDistribution(Variable(action), mu, sigma, pi)
    action = torch.clamp(action, -1.0, 1.0)
    log_prob = (prob+1e-6).log()
    entropy = 0.5*((2*pi.expand_as(sigma)*sigma).log()+1)
    return action, log_prob, entropy, mu, value
    """return {"action" : action,
            "log_prob" : log_prob,
            "entropy" : entropy,
            "mu" : mu,
            "value" : value}"""

def train(proces_number, params, shared_model, shared_optimizer):
    env = gym.make(params.env_name)
    state = env.reset()
    state = torch.from_numpy(state).float()
    
    env.seed(params.seed+proces_number)
    torch.manual_seed(params.seed + proces_number)
    
    model = Network(env.observation_space.shape[0], env.action_space, params.device).to(params.device)
    model.train()
    done = True
    length = 0
    
    while True:
        length += 1
        model.load_state_dict(shared_model.state_dict())
        log_probs = []
        entropys = []
        values = []
        rewards = []
        for i in range(params.n_steps):
            if done:
                hx = Variable(torch.zeros(1,128))
                cx = Variable(torch.zeros(1,128))
                length = 0
            else:
                hx = Variable(hx.data)
                cx = Variable(cx.data)
            value, mu , sigma, (hx, cx) = model((Variable(state), (hx, cx)))
            action, log_prob, entropy, mu, value = CalculateDistribution(value, mu, sigma)
            log_probs.append(log_prob)
            values.append(value)
            entropys.append(entropy)
            state, reward, done, _ = env.step(action.numpy()[0])
            state = torch.from_numpy(state).float()
            reward = max(min(reward, 1), -1)
            rewards.append(reward)
            done = (done or length >= params.max_episode_length)
            if done:
                state = env.reset()
                state = torch.from_numpy(state).float()
                break
        R = torch.zeros(1,1)
        if not done:
            value, _, _, _ = model((Variable(state), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        value_loss = 0
        policy_loss = 0
        R = Variable(R)
        gae = torch.zeros(1,1)
        for i in reversed(range(len(rewards))):
            # value loss
            R = params.gamma*R + rewards[i]
            A = R - values[i]
            value_loss = value_loss + 0.5*A.pow(2)
            #policy_loss
            TD_error = rewards[i] + params.gamma*values[i+1].data - values[i].data
            gae = params.tau*params.gamma*gae + TD_error
            policy_loss = policy_loss - log_probs[i].sum()*Variable(gae) - 0.01*entropys[i].sum()
        
        model.zero_grad()
        (value_loss + 0.5*policy_loss).backward()
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                break
            shared_param._grad = param.grad
        shared_optimizer.step()