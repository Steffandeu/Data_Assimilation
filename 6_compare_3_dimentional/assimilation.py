import sys
import numpy as np
from lorenz96 import Lorenz96


class Kalman(object):
    def __init__(self, true_path="save_data/data", noise_path="noise_data/data"):
        self.file_num = 1460
        self.N = 40
        self.true_path = true_path
        self.noise_path = noise_path
        self.init_true, self.init_noise = self.load_data(0)
        self.model = Lorenz96(N_dim=self.N, F=8, init_x=self.init_true)

        # make data set
        ## self.true_set : system trandition
        ## self.obs_set : observation data
        ## self.matrix_set : system trandition matrix
        ## self.noise_set : system noise
        data = self.load_data(0)
        self.true_set = np.array([data[0]])
        self.obs_set = np.array([data[1]])

        for i in range(1, self.file_num):
            data = self.load_data(i)
            true = np.array([data[0]])
            self.true_set = np.append(self.true_set, true, axis=0)
            obs = np.array([data[1]])
            self.obs_set = np.append(self.obs_set, obs, axis=0)

        # compute system trandition matrix/noise
        self.matrix_set = np.zeros([self.file_num-1, self.N, self.N])
        self.noise_set = np.zeros([self.file_num-1, self.N])

        for i in range(self.file_num-1):
            self.noise_set[i] = np.random.randn(self.N)
            x1 = np.array([self.true_set[i+1]])
            x0 = np.array([self.true_set[i]])
            x0_inv = np.linalg.pinv(x0)
            self.matrix_set[i] = np.dot(x0_inv, x1-self.noise_set[i])


    def make_covariance(self, x0, x1):
        x0_mean = x0.sum()/self.N
        x1_mean = x1.sum()/self.N

        P = np.zeros(self.N, self.N)
        for i in range(self.N):
            for j in range(self.N):
                value = (x0[i] - x0_mean) * (x1[j] - x0_mean)
                P[i][j] = value

        return P


    def noise_variance(self):
        noise_system = np.randn(self.N)
        noise_observation = np.randn(self.N)
        Q = self.make_covariance(noise_system, noise_system.transpose())
        R = self.make_covariance(noise_observation, noise_observation.transpose())

        return Q, R

    def kalman_filter(self, x, t):
        # Predict
        F = self.matrix_set[t]
        noise = np.random.randn(self.N, self.N)
        x_predict = np.dot(self.F_matrix[t], x) + self.noise_set[t]
        P = self.make_covariance(x, x.transpose())
        P = np.dot(np.dot(F, P), F.transpose()) + self.noise_variance()[1]

        # Update
        R = self.noise_variance()[1]
        y = self.obs_set[t]
        e = y - x_predict
        S = R + P
        K = np.dot(P, S.transpose())
        x = x_predict + np.dot(K, e)
        P_next = np.dot(np.indetity(self.N) - K, P)

        return x, P_next


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

        return true[0], observed[0]




if __name__ == "__main__":
    model = Kalman(true_path="save_data/data", noise_path="noise_data/data")
