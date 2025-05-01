import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from env2 import GameState

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction #aslinya tuh tanh
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

def evaluate_policy(channel_gain,state, env, agent, turns = 3):
    env = GameState(7,3)   
    total_scores = 0
    total_data_rate = 0
    total_power = 0
    total_EE=0
    dr1=0
    dr2=0
    dr3=0
    dr4=0
    dr5=0
   
    for j in range(turns):
        #s, info = env.ini()
        done = False
        MAX_STEPS = 1  # Batas maksimum langkah per episode
        step_count = 0
        a=np.zeros(5)
        while not done:
            step_count += 1
            
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True) #aslinya True
            
            print(a)
            
            next_loc= env.generate_positions() #lokasi untuk s_t
            next_channel_gain=env.generate_channel_gain(next_loc) #channel gain untuk s_t
            s_next, r, dw, tr, info = env.step(a,channel_gain,next_channel_gain)
            
            if step_count==MAX_STEPS:
                tr=True
            done = (dw or tr)
            dr1 +=info['data_rate1']
            dr2 +=info['data_rate2']
            dr3 +=info['data_rate3']
            dr4 +=info['data_rate4']
            dr5 +=info['data_rate5']
            total_scores += r
            total_EE     += info['EE']
            total_power  += info['total_power']
            
            state = s_next
            channel_gain=next_channel_gain
    return int(total_scores/turns), total_EE/turns,total_power/turns,dr1/turns,dr2/turns,dr3/turns,dr4/turns,dr5/turns

#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
