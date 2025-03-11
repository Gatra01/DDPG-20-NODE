import numpy as np 
from typing import Optional
class GameState:
    def __init__(self, nodes, p_max):
        self.nodes=nodes
        self.p_max=p_max
        self.gamma=0.01
        self.beta=1
        self.noise_power=0.01
        self.observation_space=2*nodes + nodes*nodes + 1
        self.action_space=nodes
        self.p=np.random.uniform(0, self.p_max, size=self.nodes)
    def ini(self,ini_gain,*, seed: Optional[int] = None, options: Optional[dict] = None):
        power = self.p
        #super().ini(seed=seed)
        #ini_gain= self.generate_channel_gain()
        ini_sinr=self.hitung_sinr(ini_gain,power)
        ini_data_rate=self.hitung_data_rate(ini_sinr)
        ini_EE=self.hitung_efisiensi_energi(self.p,ini_data_rate)
        
        result_array = np.concatenate((np.array(ini_data_rate), np.array([ini_EE]),np.array(power),ini_gain.flatten()))
        return result_array ,{}
        #tambahin channel gain, disamain kaya algoritma GNN
    def generate_channel_gain(self):
        channel_gain = np.random.rayleigh(scale=1, size=(self.nodes, self.nodes))
        return channel_gain
    
    def hitung_sinr(self, channel_gain, power):
        sinr=np.zeros(self.nodes)
        for node_idx in range(self.nodes):
            sinr_numerator = (abs(channel_gain[node_idx][node_idx]) ** 2) * power[node_idx]
            sinr_denominator = self.noise_power + np.sum([(abs(channel_gain[node_idx][i]) ** 2) * power[i] for i in range(self.nodes) if i != node_idx])
            sinr[node_idx] = sinr_numerator / sinr_denominator
        return sinr 
    
    def hitung_data_rate(self, sinr):
        """Menghitung data rate berdasarkan SINR"""
        for i in range(len(sinr)):
            if sinr[i]<0:
                sinr[i]=0
        data_rate = np.log(1 + sinr)
        return data_rate
    def hitung_efisiensi_energi(self,power,data_rate):
        """Menghitung efisiensi energi total"""
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        energi_efisiensi=total_rate / total_power if total_power > 0 else 0
        return energi_efisiensi

    def step(self,power,channel_gain):
        #self.last_power=power
        #new_channel_gain=self.generate_channel_gain()
        new_sinr=self.hitung_sinr(channel_gain,power)
        new_data_rate=self.hitung_data_rate(new_sinr)
        EE=self.hitung_efisiensi_energi(power,new_data_rate)
        total_daya=np.sum(power)
        result_array = np.concatenate((np.array(new_data_rate), np.array([EE]),np.array(power),channel_gain.flatten()))
        fairness = np.var(new_data_rate)  # Variansi untuk mengukur kesenjangan data rate
        reward = ( 3 * EE + np.sum(((np.array(new_data_rate)-self.gamma)*3)) -2 * fairness- 2 * np.sum(power[power <= 0])+2*(self.p_max-total_daya)  )
        #reward = 5*EE+np.sum(((np.array(new_data_rate)-self.gamma)*10).tolist())+ 5*(self.p_max-total_daya) 
        #for i in power :
        #    if i<=0:
        #        reward-=8*i

        return result_array,reward, False,False,{}
    
    
