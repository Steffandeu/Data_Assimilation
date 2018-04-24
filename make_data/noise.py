import os
import numpy as np


file_num = 1460
N_dim = 40

value_dir = "true_value/"
new_dir_path = "observation_data/"


if __name__ == "__main__":

    # Make directory for noised data
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for i in range(file_num):
        filename = "data"+str(i)+".dat"

        noise = np.random.randn(N_dim)

        with open(value_dir+filename, "r") as f:
            raw = f.readlines()

        # Raw data process
        data = np.array([raw]).astype(float)[0]
        noise_in = data + noise
        print(noise)
        print("")

        # save files
        value = noise_in
        filename = new_dir_path+"/data"+str(i)+".dat"
        np.savetxt(filename, value)

    print(data)
    print(value - data)
