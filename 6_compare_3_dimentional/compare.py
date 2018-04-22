import numpy as np
from assimilation import Kalman
from three_dimentional import Three_dimentional


true_path = "save_data/data"
noise_path = "noise_data/data"

kalman_model = Kalman(true_path=true_path,noise_path=noise_path)
three_d_model = Three_dimentional(true_path=true_path,noise_path=noise_path)
