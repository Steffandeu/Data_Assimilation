import sys
import numpy as np
from lorenz96 import Lorenz96


class EnsembleKalman(object):
    def __init__(self, true_path="make_data/true_value/data",
                        noise_path="make_data/observation_data/data"):
        self.file_num = 1460
        self.dt = 0.05   # 6 hours
        self.N = 40
        self.true_path = true_path
        self.noise_path = noise_path
        self.init_true, self.init_noise = self.load_data(0)
        self.init_P = np.zeros([self.N, self.N])
        self.model = Lorenz96(N_dim=self.N, F=8, init_x=self.init_true)
        self.ensemble_num = 64
        self.ensemble_sigma = 0.01



    def EnKF_predict(self, x, t):
        # Predict
        # Make ensemble
        ensemble = np.tile(x, (self.ensemble_num, 1))
        ensemble += self.ensemble_sigma * np.random.randn(self.N, self.ensemble_num).transpose()

        for i in range(self.ensemble_num):
            ensemble[i] = x + self.model._lorenz(ensemble[i], t) * self.dt

        return ensemble


    def EnKF_update(self, x, t):
        # update
        true, obs = self.load_data(t)
        # Make ensemble
        ensemble = np.tile(x, (self.ensemble_num, 1))
        ensemble += self.ensemble_sigma * np.random.randn(self.N, self.ensemble_num).transpose()

        # error covariance
        ensemble_x = ensemble # keep the same shape
        for i in range(self.ensemble_num):
            ensemble_x[i] = ensemble[i] - ensemble[i].mean()

        # error of output covariance
        ensemble = ensemble + np.random.randn(self.ensemble_num, self.ensemble_num) * self.ensemble_sigma
        for i in range(self.ensemble_num):
            ensemble_y[i] = ensemble[i] - ensemble[i].mean()

        U = np.dot(ensemble_x, ensemble_x.transpose()) / (self.ensemble_num - 1)
        V = np.dot(ensemble_y, ensemble_y.transpose()) / (self.ensemble_num - 1)

        H = np.dot(U, np.linalg.inv(V))

        for i in range(self.ensemble_num):
            err = obs - ensemble_y[i]
            ensemble[i] = ensemble_x[i] + np.dot(H, err.transpose()).transpose()

        return ensemble


    def make_covariance(self, x0, x1):
        x0_mean = x0.sum()/self.N
        x1_mean = x1.sum()/self.N

        P = np.zeros([self.N, self.N])
        for i in range(self.N):
            for j in range(self.N):
                value = (x0[i][0] - x0_mean) * (x1[0][i] - x1_mean)
                P[i][j] = value

        return P


    def noise_variance(self):
        noise_system = np.random.randn(self.N)
        noise_observation = np.random.randn(self.N)
        Q = self.make_covariance(noise_system, noise_system.transpose())
        R = self.make_covariance(noise_observation, noise_observation.transpose())

        return Q, R


    def load_data(self, Nth_data):
        if Nth_data > self.file_num:
            print("Not Found Such Data")
            sys.exit()

        with open(self.true_path+str(Nth_data)+".dat", "r") as f:
            true = f.readlines()
            true = np.array([true]).astype(float)

        with open(self.noise_path+str(Nth_data)+".dat", "r") as g:
            observed = g.readlines()
            observed = np.array([observed]).astype(float)

        return np.array([true[0]]), np.array([observed[0]])


    def transition(self, Nth):
        # compute system trandition matrix/noise
        data, obs_data = self.load_data(Nth)
        data_next, obs_next = self.load_data(Nth+1)

        matrix = np.dot(data_next.transpose(), np.linalg.pinv(data.transpose()))

        return matrix


    def make_covariance(self, x0, x1):
        x0_mean = x0.sum()/self.N
        x1_mean = x1.sum()/self.N

        P = np.zeros([self.N, self.N])
        for i in range(self.N):
            for j in range(self.N):
                value = (x0[i][0] - x0_mean) * (x1[0][i] - x1_mean)
                P[i][j] = value

        return P



if __name__ == "__main__":
    model = EnsembleKalman(true_path="make_data/true_value/data",noise_path="make_data/observation_data/data")

    x = np.random.randn(1, model.N)
    print(model.EnKF_predict(x, 30)[4].shape)
