from agent import DQN_agent
import numpy as np
from skimage import transform
import torch
from torch.autograd import Variable
from vizdoom import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cx = Variable(torch.zeros(64, 40, device=device)) # the cell states of the LSTM are reinitialized to zero
hx = Variable(torch.zeros(64, 40, device=device)) # the hidden states of the LSTM are reinitialized to 

left = [1, 0, 0, 0, 0, 0, 0]
right = [0, 1, 0, 0, 0, 0, 0]
shoot = [0, 0, 1, 0, 0, 0, 0]
actions = [left, right, shoot]

class params():
    batch_size = 64
    buffer_size = 10000
    learning_rate = 0.0001
    gamma = 0.99
    learnig_interval = 5
    min_update = 1e-5
    tau = 0.01
    n_steps = 3

def preprocess_frame(frame):
    cropped_frame = frame[30:-10,30:-30]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])
    return preprocessed_frame

def healthReward(health_delta):
    if health_delta == 0:
        return 0
    if health_delta < 0:
        return -0.1

def ammoReward(ammo_delta):
    if ammo_delta == 0:
        return 0
    if ammo_delta < 0:
        return -0.2

def prepopulate_buffer(game, agent):
    print("prepopulation start")
    samples_amount=0
    while True:
        game.new_episode()
        last_total_health = 100
        last_ammo = game.get_game_variable(GameVariable.AMMO2)
        next_state = state = game.get_state()
        while True:
            if samples_amount==10000:
                break
            if samples_amount %500 == 0:
                print(samples_amount)
            starting_state = state.screen_buffer
            discounted_reward = 0
            for i in range(params.n_steps):
                state = next_state
                img = state.screen_buffer
                img = preprocess_frame(img)
                
                health_delta = game.get_game_variable(GameVariable.HEALTH) - last_total_health
                last_total_health = game.get_game_variable(GameVariable.HEALTH)
        
                ammo_delta = game.get_game_variable(GameVariable.AMMO2) - last_ammo
                last_ammo = game.get_game_variable(GameVariable.AMMO2)
                
                selected_action = agent.select_action(img, 1.0, (hx[0].view(1,-1), cx[0].view(1,-1)))
                if i == 0:
                    first_action = selected_action
                action = actions[selected_action]
                reward = game.make_action(action)
                discounted_reward += (params.gamma**i)*(reward+healthReward(health_delta)+ammoReward(ammo_delta))
                done = game.is_episode_finished()
                if done:
                    break
                next_state = game.get_state()
            if done:
                next_img = np.zeros((84, 84), dtype='uint8')
                agent.memoryBuffer.add(preprocess_frame(starting_state), first_action, discounted_reward, next_img, done)
                break
            next_img = next_state.screen_buffer
            next_img = preprocess_frame(next_img)
            agent.memoryBuffer.add(preprocess_frame(starting_state), first_action, discounted_reward, next_img, done)

            samples_amount+=1
        if samples_amount==10000:
            break
    print("prepopulation stop")