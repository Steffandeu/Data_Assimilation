import numpy as np
import matplotlib.pyplot as plt

from KF import Kalman
from three_dimentional import ThreeDVar



def processs(name):
    if name == "kalman":
        model = Kalman()
    else:
        model = ThreeDVar()

    file_num = 50
    obs_buffer = np.zeros(file_num)
    true_buffer = np.zeros(file_num)
    x_buffer = np.zeros(file_num)
    norm = np.zeros(file_num)

    true, obs = model.load_data(0)
    x = true
    P = np.identity(model.N)

    for t in range(file_num):
        print("step : ",t)
        true, obs = model.load_data(t)
        obs_buffer[t] = obs[0][0]
        true_buffer[t] = true[0][0]

        # Predict
        x, P = model.predict(x, P, t)
        print(P)
        x_buffer[t] = x[0][0]

        # Update
        x, P = model.update(x, P, t)

        # Stock Norm
        norm[t] = np.linalg.norm(x[0]-true[0])

    return x_buffer, obs_buffer, true_buffer, norm


if __name__ == "__main__":

    name = "three"

    save = False

    x_t, obs_t, true_t, norm_t = processs(name)

    name = "kalman"

    x_k, obs_k, true_k, norm_k = processs(name)

    #print(x.shape)

    plt.figure(figsize=(20,3))

    plt.subplot(131)
    plt.plot(obs_t, ".", label="Observation")
    plt.plot(true_t, "r",label="Real")
    plt.plot(x_t, "y",label="Assimilation")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("3D Var Data Assimilation")

    plt.subplot(132)
    plt.plot(obs_k, ".", label="Observation")
    plt.plot(true_k, "r",label="Real")
    plt.plot(x_k, "y",label="Assimilation")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("Kalman Filter Data Assimilation")

    plt.subplot(133)
    plt.plot(norm_t, "r",label="3D")
    plt.plot(norm_k, "b",label="Kalman")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("L2 Norm")

    if save:
        plt.savefig("./images/multi.png")

    plt.show()
