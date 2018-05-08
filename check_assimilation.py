import numpy as np
import matplotlib.pyplot as plt

from lorenz96 import Lorenz96
from KF import Kalman
from three_dimentional import Three_dimentional as Three


model_k = Kalman(true_path="make_data/true_value/data",
                 noise_path="make_data/observation_data/data")

model_t = Three(true_path="make_data/true_value/data",
                noise_path="make_data/observation_data/data")

model = model_t

t_len = 80


def kalman():
    title = "Kalman Filter"
    # observation data : first
    obs_buffer = np.zeros(1)
    pred_buffer = np.zeros(1)
    error_buffer = np.zeros(1)
    for t in range(t_len):
        true, obs = model_k.load_data(t)
        predict = model_k.KF_update(true, t)[0]
        error_buffer = np.append(error_buffer, np.linalg.norm(predict-obs))
        obs_buffer = np.append(obs_buffer, obs[0][0])
        pred_buffer = np.append(pred_buffer, predict[0][0])
    obs_buffer = obs_buffer[1:]
    pred_buffer = pred_buffer[1:]
    error_buffer = error_buffer[1:]

    return obs_buffer, pred_buffer, error_buffer, title


def three(A):
    title = "Three Dimentional Variance"
    # observation data : first
    obs_buffer = np.zeros(1)
    pred_buffer = np.zeros(1)
    error_buffer = np.zeros(1)
    for t in range(t_len):
        true, obs = model_t.load_data(t)
        predict = model_t.update(true, t, A)[0]
        error_buffer = np.append(error_buffer, np.linalg.norm(predict-obs))
        obs_buffer = np.append(obs_buffer, obs[0][0])
        pred_buffer = np.append(pred_buffer, predict[0][0])
    obs_buffer = obs_buffer[1:]
    pred_buffer = pred_buffer[1:]
    error_buffer = error_buffer[1:]

    return obs_buffer, pred_buffer, error_buffer, title


# set plot
_, _, error_k, _ = kalman()
_, _, error_t, _ = three(0.01)
time = np.arange(0, t_len) / 5.0
plt.plot(time, error_k, label="Kalman Filter")
plt.plot(time, error_t, label="Three Dimentional Variance")
plt.xlabel("time")
plt.ylabel("value")
plt.title("Error in Data Assimilation")
plt.legend()
plt.savefig("images/error_in_assimilation.png")
plt.show()
