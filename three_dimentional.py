import sys
import numpy as np
from lorenz96 import Lorenz96


class ThreeDVar(object):
    def __init__(self, true_path="make_data/true_value/data",
                 noise_path="make_data/observation_data/data"):
        self.file_num = 1460
        self.N = 40   # dimention
        self.dt = 10e-2 # 6 hours
        self.true_path = true_path
        self.noise_path = noise_path
        self.Q = np.identity(self.N)
        self.R = np.identity(self.N)
        data, _ = self.load_data(0)
        self.model = Lorenz96(N_dim=self.N, F=8, init_x=data)


    def predict(self, x_a, P_a, t):
        # input : x -> 1x40
        #         P -> 40x40
        dx = self.model._lorenz(x_a[0], t)
        dx = np.array([dx])
        F = np.identity(len(x_a[0]))
        G = np.identity(len(x_a[0]))
        #print("\n\ndx : \n",0.01*dx,"\n\n")
        x_f = F.dot(x_a.transpose()).transpose() + self.dt * dx
        noise = G.dot(self.Q).dot(G.transpose())
        P_f = F.dot(P_a).dot(F.transpose()) + noise

        return x_f, P_f


    def update(self, x_f, P_f, t):
        H = np.identity(self.N) # 40x40
        I = np.identity(self.N)
        y = self.load_data(t)[1] # 1x40
        e = y - H.dot(x_f.transpose()).transpose()
        S = self.R + H.dot(P_f).dot(H.transpose())
        K = P_f.dot(H.transpose()).dot(np.linalg.inv(S))
        x_a = x_f + K.dot(e.transpose()).transpose()
        P_a = (I - K.dot(H)).dot(P_f)

        return x_a, P_a


    def noise_variance(self):
        q_noise = np.random.randn(self.N)
        r_noise = np.random.randn(self.N)
        q_mean = q_noise.mean()
        r_mean = r_noise.mean()

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



if __name__ == "__main__":
    model = Kalman(true_path="make_data/true_value/data", noise_path="make_data/observation_data/data")

    x = np.random.randn(1, model.N)
    P = np.random.randn(model.N, model.N)
    print(x.transpose().shape)
    print("predict x : ", model.KF_predict(x, P, 10)[0].shape)
    print("update x : ", model.KF_update(x, P, 10)[0].shape)

    print("load data : ",model.load_data(4)[0].shape)
