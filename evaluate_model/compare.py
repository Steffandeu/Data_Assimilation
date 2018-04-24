import numpy as np
from assimilation import Kalman
from three_dimentional import Three_dimentional


true_path = "save_data/data"
noise_path = "noise_data/data"


if __name__ == "__main__":
    kalman_model = Kalman(true_path=true_path,noise_path=noise_path)
    three_d_model = Three_dimentional(true_path=true_path,noise_path=noise_path)

    # t = 0
    init_data = kalman_model.true_set[0]
    init_obs = kalman_model.obs_set[0]

    a, b = kalman_model.kalman_filter(init_data, 0)
    x, y = three_d_model.three_d(init_data, 0, 0)

    print(a)
    print(y)
