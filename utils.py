import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, 128)
        self.l3 = nn.Linear(128, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.sigmoid(self.l3(a)) * self.maxaction #aslinya tuh tanh
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
    total_scores = 0
    total_data_rate = 0
    total_power = 0
    total_EE=0
   
    for j in range(turns):
        #s, info = env.ini()
        done = False
        MAX_STEPS = 200  # Batas maksimum langkah per episode
        step_count = 0
        while not done:
            step_count += 1
            # Take deterministic actions at test time
            a = agent.select_action(state, deterministic=True) #aslinya True
            #if render :
            print(a)
            s_next, r, dw, tr, info = env.step(a,channel_gain)
            
            if step_count==MAX_STEPS:
                tr=True
            done = (dw or tr)

            total_scores += r
            total_power += info.get('power', 0)
            total_data_rate += info.get('rate', 0)
            total_EE+=info.get('EE',0)
            s = s_next
       
    return int(total_scores/turns), total_EE/turns


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
