
import gym
import time
import torch

import numpy as np

from torch.autograd import Variable
from collections import deque
from model import Network

def test(params, shared_model):
    env = gym.make(params.env_name)
    env.seed(params.seed)
    torch.manual_seed(params.seed)
    last_rewards = deque(maxlen=100)
    model = Network(env.observation_space.shape[0], env.action_space, params.device).to(params.device)
    model.eval()
    state = env.reset()
    episode = 0
    max_reward = -10000000
    done = True
    length = 0
    rewards = 0
    with torch.no_grad():
        while True:
            length+=1
            if done:
                episode += 1
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 128))
                hx = Variable(torch.zeros(1, 128))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)
            state = torch.from_numpy(state).float().to(params.device)
            value, mu , sigma, (hx, cx) = model((Variable(state), (hx, cx)))
            mu = torch.clamp(mu.data, -1.0, 1.0)
            state, reward, done, _ = env.step(mu.numpy()[0])
            rewards += reward
            #env.render()
            done = done or length >=params.max_episode_length
            if done:
                state = env.reset()
                last_rewards.append(rewards)
                print(episode,"|reward:", rewards,"|last:",np.mean(last_rewards),"|max_reward:",max_reward, "|ts:",length)
                if rewards > max_reward:
                    max_reward = rewards
                    print("   NEW MAX saving model   :", max_reward)
                    file_name = 'checkpoint_final.pth'
                    torch.save(model.state_dict(), file_name)
                if np.mean(last_rewards) >=params.solved_score:
                    print("Solved in episode ", episode, " with mean " , np.mean(last_rewards))
                    file_name = 'checkpoint_final.pth'
                    torch.save(model.state_dict(), file_name)
                    break
                rewards = 0
                length = 0
                time.sleep(60)