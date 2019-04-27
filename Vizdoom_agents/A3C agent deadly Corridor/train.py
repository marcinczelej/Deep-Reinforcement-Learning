
from model import a3cNetwork
import torch
from torch.autograd import Variable
from vizdoom import *
from utility import *

def train(process_number, params, shared_model, shared_optimizer):
    game = DoomGame()
    # change this for vizdoom defend_the_center path
    game.load_config("./scenario/deadly_corridor.cfg")
    game.set_screen_format(ScreenFormat.GRAY8)
    game.add_available_game_variable(GameVariable.KILLCOUNT)
    game.add_available_game_variable(GameVariable.HEALTH)
    game.set_window_visible(False)
    game.init()
    
    torch.manual_seed(params.seed + process_number)
    
    game.new_episode()
    last_total_health = 100
    last_total_kills = 0
    state = game.get_state()
    img = state.screen_buffer
    img = preprocess_frame(img).to(params.device)
    
    done = True
    length = 0
    
    model = a3cNetwork(params.action_space).to(params.device)
    model.train()
    
    while True:
        model.load_state_dict(shared_model.state_dict())
        values = []     # TD part, advantage part
        rewards = []    # to collect rewards ( reinforce + TD part)
        entropys = []   # to make use of entropy
        log_probs = []  # needed for reinforce part
        
        for i in range(params.n_steps):
            if done:
                length = 0
                hx = Variable(torch.zeros(1, 128, device=params.device))
                cx = Variable(torch.zeros(1, 128, device=params.device))
            else:
                length +=1
                hx = Variable(hx.data)
                cx = Variable(cx.data)
            actor_return, value, (hx, cx) = model((Variable(img), (hx, cx)))
            values.append(value)
            entropys.append(actor_return["entropy"]) 
            selected_action = actor_return["action"].data.numpy()[0][0]
            log_probs.append(actor_return["log_prob"].gather(1, actor_return["action"]))
            action = actions[selected_action]
            reward = game.make_action(action)
            health_delta = game.get_game_variable(GameVariable.HEALTH) - last_total_health
            last_total_health = game.get_game_variable(GameVariable.HEALTH)
            
            kills_delta = game.get_game_variable(GameVariable.KILLCOUNT) - last_total_kills
            last_total_kills = game.get_game_variable(GameVariable.KILLCOUNT)
            
            rewards.append(reward+healthReward(health_delta)+killsReward(kills_delta))
            done = game.is_episode_finished() or length >=4200
            if done:
                game.new_episode()
                state = game.get_state()
                img = state.screen_buffer
                img = preprocess_frame(img)
                last_total_health = 100
                last_total_kills = 0
                break
        R = torch.zeros(1,1)
        if not done:
            state = game.get_state()
            img = state.screen_buffer
            img = preprocess_frame(img).to(params.device)
            _, value, _ = model((Variable(img), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1,1)
        # GAE A = Q - V
        for i in reversed(range(len(rewards))):
            #value loss
            R = rewards[i] + R*params.gamma
            advantage_i = R - values[i]
            value_loss = value_loss + 0.5*advantage_i.pow(2)
            #policy loss
            TD = rewards[i] + params.gamma*values[i+1].data - values[i].data
            gae = gae*params.gamma*params.tau + TD
            policy_loss = policy_loss - log_probs[i]*Variable(gae) - 0.01*entropys[i]
        loss = policy_loss + 0.5*value_loss
        model.zero_grad()
        loss.backward()
        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is not None:
                break
            shared_param._grad = param.grad
        shared_optimizer.step()
