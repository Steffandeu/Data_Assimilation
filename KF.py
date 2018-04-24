import sys
import numpy as np
from lorenz96 import Lorenz96


class Kalman(object):
    def __init__(self, true_path="make_data/true_value/data", noise_path="make_data/observation_data/data"):
        self.file_num = 1460
        self.N = 40
        self.true_path = true_path
        self.noise_path = noise_path
        self.init_true, self.init_noise = self.load_data(0)
        self.init_P = np.zeros([self.N, self.N])
        self.model = Lorenz96(N_dim=self.N, F=8, init_x=self.init_true)


    def KF_predict(self, x, t):
        # Predict
        F = self.transition(t) # 40x40
        noise = np.random.randn(self.N, 1) # 40x1
        x_predict = np.dot(F, x.transpose()) + noise # 40x1

        error = x - self.load_data(t)[0] # 1x40
        P = self.make_covariance(error.transpose(), error) # 40x40
        P = np.dot(np.dot(F, P), F.transpose()) + np.random.randn(self.N, self.N)

        return x_predict.transpose(), P


    def KF_update(self, x, t):
        # Update
        x_predict, P_predict = self.KF_predict(x, t)
        x, y = self.load_data(t)
        e = y - x_predict # 1x40
        S = np.random.randn(self.N, self.N) + P_predict # 40x40
        K = np.dot(P_predict, np.linalg.inv(S)) # 40x40
        x = x_predict + np.dot(K, e.transpose()) # 40x1
        P_next = np.dot(np.identity(self.N) - K, P_predict)

        return x.transpose(), P_next



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



if __name__ == "__main__":
    model = Kalman(true_path="save_data/data", noise_path="noise_data/data")

    x = np.random.randn(1, model.N)
    print(x.transpose().shape)

    print(model.KF_predict(x, 10))
    print()
    print(model.KF_update(x, 10))
