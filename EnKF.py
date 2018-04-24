import sys
import numpy as np
from lorenz96 import Lorenz96


class EnsembleKalman(object):
    def __init__(self, true_path="save_data/data", noise_path="noise_data/data"):
        self.file_num = 1460
        self.dt = 0.05
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
        print(ensemble.shape)

        for i in range(self.ensemble_num):
            ensemble[i] = x + self.model._lorenz(ensemble[i], t) * self.dt

        return ensemble


    def EnKF_filter(self, x, t):
        # filter
        x_predict = self.EnKF_predict(x, t)
        x_filter = x_predict - x_predict.mean()

        ensemble = EnKF_predict(x, t)

        V = np.zeros([self.N, self.N])
        for i in range(self.ensemble_num):
            V += self.make_covariance(ensemble[i].transpose(), ensemble[i])
        V /= (self.ensemble_num - 1)

        R = np.random.randn(self.N, self.N)
        K = np.dot(V, np.linalg.inv(V + R))

        for i in range(self.ensemble_num):
            #ensemble[i] = x + self.model._lorenz(ensemble[i], t) * self.dt


    #def EnKF_smooth(self):
        # smooth



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
    model = EnsembleKalman(true_path="save_data/data", noise_path="noise_data/data")

    x = np.random.randn(1, model.N)
    print(model.EnKF_predict(x, 30).shape)
