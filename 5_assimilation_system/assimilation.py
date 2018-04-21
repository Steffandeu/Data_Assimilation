import sys
import numpy as np


class Kalman(object):
    def __init__(self, true_path="save_data/data", noise_path="noise_data/data"):
        self.file_num = 1460
        self.true_path = true_path
        self.noise_path = noise_path


    def load_data(self, Nth_data):
        if Nth_data > self.file_num:
            print("Not Found Such Data")
            sys.exit()

        with open(true_path+str(Nth_data-1)+".dat", "r") as f:
            true = f.readlines()
            true = np.array([true]).astype(float)

        with open(noise_path+str(Nth_data-1)+".dat", "r") as g:
            observed = g.readlines()
            observed = np.array([observed]).astype(float)

        return true[0], observed[0]


if __name__ == "__main__":
    model = Kalman(true_path=true_path, noise_path=noise_path)

    a, b = model.load_data(100)

    print(a)
    print(b)
    print(a - b)
