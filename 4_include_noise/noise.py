import os
import numpy as np


file_num = 1460
N_dim = 40

new_dir_path = "noise_data"


if __name__ == "__main__":

    # Make directory for noised data
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for i in range(file_num):
        filename = "data"+str(i)+".dat"

        # Set noise
        np.random.seed()
        noise = np.random.uniform(0, 1.0, N_dim)

        with open("data/"+filename, "r") as f:
            raw = f.readlines()

        # Raw data process
        data = np.array([raw]).astype(float)
        noise_in = data + noise

        # save files
        np.savetxt(new_dir_path + "/" + filename, noise_in)
