import numpy as np 
from typing import Optional

class GameState:
    def __init__(self, nodes, p_max, area_size=(5, 5)):
        self.nodes = nodes
        self.p_max = p_max
        self.gamma = 0.01
        self.beta = 1
        self.noise_power = 0.01
        self.area_size = area_size
        self.positions = self.generate_positions()
        self.observation_space = 2 * nodes * nodes + nodes  # data_rate, power, channel gain, EE
        self.action_space = nodes
        self.p = np.random.uniform(0, self.p_max, size=self.nodes)
    def reset(self,gain,*, seed: Optional[int] = None, options: Optional[dict] = None):
        power = self.p
        #super().ini(seed=seed)
        #loc = self.generate_positions()
        #gain= self.generate_channel_gain(loc)
        intr=self.interferensi(power,gain)
        #ini_sinr=self.hitung_sinr(ini_gain,intr,power)
        #ini_data_rate=self.hitung_data_rate(ini_sinr)
        #ini_EE=self.hitung_efisiensi_energi(self.p,ini_data_rate)
        gain_norm=norm(gain)
        intr_norm = norm(intr)
        p_norm=norm(power)
        
        result_array = np.concatenate((np.array(gain_norm).flatten(), np.array(intr_norm).flatten(),np.array(p_norm)))
        return result_array ,{}

    def step(self,power,channel_gain):
        new_intr=self.interferensi(power,channel_gain)
        new_sinr=self.hitung_sinr(channel_gain,new_intr,power)
        new_data_rate=self.hitung_data_rate(new_sinr)
        EE=self.hitung_efisiensi_energi(power,new_data_rate)
        total_daya=np.sum(power)
        gain_norm=norm(channel_gain)
        intr_norm = norm(new_intr)
        p_norm=norm(power)
        result_array = np.concatenate((np.array(gain_norm).flatten(), np.array(intr_norm).flatten(),np.array(p_norm)))
        fairness = np.var(new_data_rate)  # Variansi untuk mengukur kesenjangan data rate
        reward = (self.p_max-total_daya) + EE
        for i in power :
            if i <= 0 :
                reward -= 0.5
        for i in new_data_rate :
            if i <= 0.05 :
                reward -= 0.5
        #reward =np.sum(((np.array(new_data_rate)-self.gamma)*10).tolist())+ 5*(self.p_max-total_daya) 
        #for i in power :
        #    if i<=0:
        #        reward-=8*i

        return result_array,reward, False,False,{'power': total_daya,'rate': np.sum(new_data_rate),'EE': EE}

    def norm(self,x):
        x_log = np.log10(x + 1e-10)  # +1e-10 untuk menghindari log(0)
        x_min = np.min(x_log)
        x_max = np.max(x_log)
        return (x_log - x_min) / (x_max - x_min + 1e-10) 

    def generate_positions(self):
        """Generate random positions for all nodes in 2D space (meter)"""
        loc = np.random.uniform(0, self.area_size[0], size=(self.nodes, self.nodes))
        for i in range (self.nodes) :
            for j in range (self.nodes):
              current = loc[i][j]
              loc[j][i]=current
        return loc
    def generate_channel_gain(self, positions):
        channel_gain = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    distance = np.linalg.norm(self.positions[i] - self.positions[j]) + 1e-6  # avoid zero
                    path_loss_dB = 128.1 + 37.6 * np.log10(distance / 1000)  # example log-distance PL
                    path_loss_linear = 10 ** (-path_loss_dB / 10)
                    rayleigh = np.random.rayleigh(scale=1)
                    channel_gain[i][j] = path_loss_linear * rayleigh
                else:
                    channel_gain[i][j] = np.random.rayleigh(scale=1)
        return channel_gain
    def interferensi(self, power,channel_gain):
        interferensi = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                if i != j:
                    interferensi[i][j] = channel_gain[i][j] * power [i]
                else:
                    interferensi[i][j] = 0
        return interferensi
    

    def hitung_sinr(self, channel_gain, interferensi, power):
        sinr = np.zeros(self.nodes)
        for node_idx in range(self.nodes):
            sinr_numerator = (abs(channel_gain[node_idx][node_idx])) * power[node_idx]
            sinr_denominator = self.noise_power**2 + np.sum([(abs(interferensi[node_idx][i])) for i in range(self.nodes) if i != node_idx])
            sinr[node_idx] = sinr_numerator / sinr_denominator
        return sinr 

    def hitung_data_rate(self, sinr):
        sinr = np.maximum(sinr, 0)  # jika ada yang negatif, dibatasi 0
        return np.log(1 + sinr)

    def hitung_efisiensi_energi(self, power, data_rate):
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        return total_rate / total_power if total
