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
        self.l2 = nn.Linear(net_width, 512)
        self.l3 = nn.Linear(512,256)
        self.l4 = nn.Linear(256, action_dim)
        

        self.maxaction = maxaction
        nn.init.zeros_(self.l4.weight)
        self.l4.bias.data.fill_(0.0)

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.relu(self.l3(a))
        logits = self.l4(a)                                 # no tanh
        probs  = torch.softmax(logits, dim=-1)              # sum=1, each >0
        return probs * self.maxaction                        # ini namanya metode soft-max head
        #a = torch.sigmoid(self.l3(x)) * self.maxaction      # kalo yang ini namanya sigmoid head    
        #return a
      


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=1024):
        super().__init__()
        # pertama‐tama embed state saja
        self.l1 = nn.Linear(state_dim, net_width)
        self.ln1 = nn.LayerNorm(net_width)

        # setelah itu concat action, lalu dua layer lagi
        self.l2 = nn.Linear(net_width + action_dim, net_width//2)
        self.ln2 = nn.LayerNorm(net_width//2)

        self.l3 = nn.Linear(net_width//2, net_width//4)
        self.ln3 = nn.LayerNorm(net_width//4)

        # output Q‐value scalar
        self.l4 = nn.Linear(net_width//4, 1)

    def forward(self, state, action):
        """
        state:  Tensor [B, state_dim]
        action: Tensor [B, action_dim]
        """
        x = F.relu(self.ln1(self.l1(state)))             # [B, net_width]
        x = torch.cat([x, action], dim=-1)               # [B, net_width+action_dim]
        x = F.relu(self.ln2(self.l2(x)))                 # [B, net_width//2]
        x = F.relu(self.ln3(self.l3(x)))                 # [B, net_width//4]
        q = self.l4(x)                                   # [B, 1]
        return q

def evaluate_policy(channel_gain,state, env, agent, turns = 3):
    env = GameState(20,5)   
    total_scores = 0
    total_data_rate = 0
    total_power = 0
    total_EE=0
   
    for j in range(turns):
        #s, info = env.ini()
        done = False
        MAX_STEPS = 1  # Batas maksimum langkah per episode
        step_count = 0
        a=np.zeros(20)
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

            total_scores += r
            state = s_next
            channel_gain=next_channel_gain
    return int(total_scores/turns)

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
