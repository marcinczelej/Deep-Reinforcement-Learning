from model import a3cNetwork
import numpy as np
import time
import torch
from vizdoom import *
from utility import *
from torch.autograd import Variable

def test(params, shared_model):
    game = DoomGame()
    # change this for vizdoom defend_the_center path
    game.load_config("./scenario/deadly_corridor.cfg")
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_window_visible(False)
    game.init()
    
    game.new_episode()
    state = game.get_state()
    img = state.screen_buffer
    img = preprocess_frame(img).to(params.device)
    
    done = True
    torch.manual_seed(0)
    model = a3cNetwork(params.action_space).to(params.device)
    model.eval()
    episode = 0
    length = 0
    max_reward = -999999999999
    rewards = 0
    with torch.no_grad():
        while True:
            length+=1
            if done:
                episode += 1
                model.load_state_dict(shared_model.state_dict())
                cx = Variable(torch.zeros(1, 256))
                hx = Variable(torch.zeros(1, 256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)
            actor_return, value, (hx, cx) = model((Variable(img), (hx, cx)))
            selected_action = actor_return["action"].data.numpy()[0][0]
            action = actions[selected_action]
            reward = game.make_action(action)
            rewards += reward
            done = game.is_episode_finished() or length >=4200
            if done:
                print(episode,"|reward:", rewards,"|max_reward:",max_reward, "|ts:",length)
                game.new_episode()
                state = game.get_state()
                img = state.screen_buffer
                img = preprocess_frame(img)
                if rewards > max_reward:
                    max_reward = rewards
                    print("   NEW MAX saving model   :", max_reward)
                    file_name = 'checkpoint_final.pth'
                    torch.save(model.state_dict(), file_name)
                if rewards >=params.solved_score:
                    print("Solved in episode ", episode)
                    file_name = 'checkpoint_final.pth'
                    torch.save(model.state_dict(), file_name)
                    break
                rewards = 0
                length = 0
                time.sleep(60)
        