from collections import deque
import gym
from model import a3cNetwork
import numpy as np
import time
import torch

def test(params, shared_model, device):
    
    env = gym.make(params.env_name)
    env.seed(0)
    torch.manual_seed(0)
    win_condition = 195
    last_rewards = deque(maxlen=100)
    model = a3cNetwork(params.state_space, params.action_space, device).to(device)
    
    state = env.reset()
    episode = 0
    max_reward = 0
    
    while True:
        model.load_state_dict(shared_model.state_dict())
        episode+=1
        rewards = 0
        length = 0
        while True:
            length+=1
            actor_return, _ = model(state)
            action = actor_return["action"].data.numpy()[0][0]
            state, reward, done, _ = env.step(action)
            rewards += reward
            env.render()
            if done or length >=200:
                state = env.reset()
                break
        last_rewards.append(rewards)
        if rewards > max_reward:
            max_reward = rewards
        
        if np.mean(last_rewards) >=195:
            print("AI won in episode ", episode, " with mean score of : ", np.mean(last_rewards))
            break
        if episode%10 == 0:
            print(episode, " reward = ", rewards, ", | means score = ", np.mean(last_rewards), " | max reward = ", max_reward)
        #time.sleep(2)