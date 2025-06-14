import numpy as np 
from scipy.spatial.distance import cdist
from typing import Optional

class GameState:
    def __init__(self, nodes, p_max, area_size=(20, 20)):
        self.nodes = nodes
        self.p_max = p_max
        self.gamma = 0.01
        self.beta = 1
        self.noise_power = 0.01
        self.area_size = area_size
        # initialize shadowing parameter, can be updated externally each episode
        self.lambda_shadow_current = 0.0
        self.positions = self.generate_positions()
        self.observation_space = 2 * nodes * nodes + nodes  # interferensi, channel gain, power
        self.action_space = nodes
        self.p = np.random.uniform(0, 3, size=self.nodes)

    def sample_valid_power(self):
        rand = np.random.rand(self.nodes)
        rand /= np.sum(rand)
        return rand * self.p_max

    def reset(self, gain, *, seed: Optional[int] = None, options: Optional[dict] = None):
        power = self.sample_valid_power()
        intr = self.interferensi(power, gain)
        gain_norm = self.norm(gain)
        intr_norm = self.norm(intr)
        p_norm = self.norm(power)
        result_array = np.concatenate((gain_norm.flatten(), intr_norm.flatten(), p_norm))
        return result_array, {}

    def step_function(self, x):
        return 1 if x > 0 else 0

    def step(self, power, channel_gain, next_channel_gain):
        intr = self.interferensi(power, channel_gain)
        next_intr = self.interferensi(power, next_channel_gain)
        sinr = self.hitung_sinr(channel_gain, intr, power)
        data_rate = self.hitung_data_rate(sinr)
        data_rate_constraint = [5 * self.step_function(2 - dr) for dr in data_rate]
        EE = self.hitung_efisiensi_energi(power, data_rate)
        total_daya = np.sum(power)
        fail_power = total_daya > self.p_max
        dw = bool(fail_power)

        info = {f'data_rate{i+1}': data_rate[i] for i in range(self.nodes)}
        info['EE'] = EE
        info['total_power'] = float(total_daya)

        reward = -np.sum(data_rate_constraint) + EE - 5 * self.step_function(total_daya - self.p_max)
        obs = np.concatenate((self.norm(next_channel_gain).ravel(), self.norm(next_intr).ravel(), self.norm(power)))
        return obs.astype(np.float32), float(reward), dw, False, info

    def norm(self, x):
        x = np.maximum(x, 1e-10)
        x_log = np.log10(x)
        x_min = np.min(x_log)
        x_max = np.max(x_log)
        return (x_log - x_min) / (x_max - x_min + 1e-10)

    def generate_positions(self, minDistance=2, subnet_radius=2, minD=0.5):
        rng = np.random.default_rng()
        bound = self.area_size[0] - 2 * subnet_radius
        X = np.zeros((self.nodes, 1), dtype=np.float64)
        Y = np.zeros((self.nodes, 1), dtype=np.float64)
        dist_2 = minDistance ** 2
        nValid = 0
        loop_counter = 0
        while nValid < self.nodes and loop_counter < 1e6:
            newX = bound * (rng.uniform() - 0.5)
            newY = bound * (rng.uniform() - 0.5)
            if all((X[:nValid] - newX)**2 + (Y[:nValid] - newY)**2 > dist_2):
                X[nValid] = newX
                Y[nValid] = newY
                nValid += 1
            loop_counter += 1
        if nValid < self.nodes:
            raise RuntimeError("Gagal menghasilkan semua controller dengan minDistance")
        X += self.area_size[0] / 2
        Y += self.area_size[0] / 2
        gwLoc = np.hstack((X, Y))
        dist_rand = rng.uniform(minD, subnet_radius, size=(self.nodes, 1))
        ang = rng.uniform(0, 2 * np.pi, size=(self.nodes, 1))
        D_X = X + dist_rand * np.cos(ang)
        D_Y = Y + dist_rand * np.sin(ang)
        dvLoc = np.hstack((D_X, D_Y))
        return cdist(gwLoc, dvLoc)

    def generate_channel_gain(self, dist, f=6e9, r=3.5, lambda_shadow_current=None):
        """
        Hitung channel gain h_mn sesuai Eq. (4) dengan lognormal shadowing dan Rayleigh fading.
        Args:
            dist: matriks [N x N] jarak antar node
            f: frekuensi carrier dalam Hz
            r: path loss exponent
            lambda_shadow_current: standar deviasi shadowing lognormal saat ini (dB)
        Returns:
            channel_gain: matriks [N x N]
        """
        if lambda_shadow_current is None:
            lambda_shadow_current = self.lambda_shadow_current
        N = dist.shape[0]
        c = 3e8
        coeff = (c**2) / (4 * np.pi * f)**2
        # Shadowing kappa ~ LogNormal(0, lambda_shadow_current^2)
        log_kappa = np.random.normal(loc=0.0, scale=lambda_shadow_current, size=(N, N))
        kappa = np.exp(log_kappa)
        # Fading zeta ~ CN(0, 1)
        real = np.random.normal(0, 1, size=(N, N))
        imag = np.random.normal(0, 1, size=(N, N))
        zeta = real + 1j * imag
        # Channel gain
        channel_gain = coeff * kappa * (np.abs(zeta)**2) / (dist ** r)
        return channel_gain

    def interferensi(self, power, channel_gain):
        interferensi = np.zeros((self.nodes, self.nodes))
        for i in range(self.nodes):
            for j in range(self.nodes):
                interferensi[i, j] = channel_gain[i, j] * power[i] if i != j else 0
        return interferensi

    def hitung_sinr(self, channel_gain, interferensi, power):
        sinr = np.zeros(self.nodes)
        for i in range(self.nodes):
            num = abs(channel_gain[i, i]) * power[i]
            den = self.noise_power + np.sum(np.abs(interferensi[i, :]))
            sinr[i] = num / den
        return sinr

    def hitung_data_rate(self, sinr):
        sinr = np.maximum(sinr, 0)
        return np.log(1 + sinr)

    def hitung_efisiensi_energi(self, power, data_rate):
        total_power = np.sum(power)
        total_rate = np.sum(data_rate)
        return total_rate / total_power if total_power > 0 else 0
