import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import numpy as np
from env2 import GameState

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super().__init__()
        # shared trunk
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.ReLU(),
            nn.LayerNorm(net_width),
            nn.Linear(net_width, net_width//2),
            nn.ReLU(),
            nn.LayerNorm(net_width//2),
            nn.Linear(net_width//2, net_width//4),
            nn.ReLU(),
            nn.LayerNorm(net_width//4),
        )
        # two heads
        self.dist_head  = nn.Linear(net_width//4, action_dim)  # for softmax
        self.scale_head = nn.Linear(net_width//4, 1)           # for budget

        self.maxaction = maxaction

    def forward(self, state):
        x = self.net(state)                   # [B, hidden]
        logits = self.dist_head(x)            # [B, action_dim]
        dist   = F.softmax(logits, dim=-1)    # sum to 1

        scale  = torch.sigmoid(self.scale_head(x)).squeeze(-1)  
        # scale in (0,1), shape [B]

        total_power = scale * self.maxaction  # shape [B]
        # expand total_power to [B,action_dim] so we can multiply
        return dist * total_power.unsqueeze(-1)


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
