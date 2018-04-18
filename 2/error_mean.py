#! /Users/komi/.pyenv/shims/python
# -*- Coding: utf-8 -*-
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import argparse

from lorenz96 import Lorenz96


# Set initial data for this model
N_dim = 40   # How many dimentions
F = 8        # Set the parameter of this model
t_end = 2    # How long to set time length
dt = 0.01    # time step
err = 1e-15  # error
tick = 0.2   # plot x-ticks

parser = argparse.ArgumentParser(
                    prog="whether to save image",
                    usage="by 'python main.py -s'",
                    description="you can save image",
                    epilog="end",
                    add_help="True"
                    )

# select CUI or GUI
parser.add_argument("-s", "--Save", dest="Save",
                    action="store_true",
                    default=False,
                    help="save file")

args = parser.parse_args()


if __name__ == "__main__":
    # Set initial parameters of x
    x = F * np.ones(N_dim)
    x[N_dim-1] += 0.01

    # Constructer
    model = Lorenz96(N_dim=N_dim, F=F, init_x=x)
    truth = model.Runge_Kutta_4(dt=dt, t_end=t_end)

    # Include error
    x = F * np.ones(N_dim)
    x[N_dim-1] += 0.01 + err

    # Constructer
    model = Lorenz96(N_dim=N_dim, F=F, init_x=x)
    error_ver = model.Runge_Kutta_4(dt=dt, t_end=t_end)

    # Evaluate
    error = error_ver - truth
    t = np.arange(0, t_end+dt, dt)
    error_set = np.array([LA.norm(error[0])])
    for i in range(0, len(error)):
        error_t = LA.norm(error[i])
        error_set = np.append(error_set, error_t)

    # plot error growth bar
    t_tick = np.arange(0, t_end+tick, tick)
    growth_set = error_set[int(1/tick)] - error_set[0]
    for i in range(1, int(t_end/tick)+1):
        growth = error_set[int((i+1)/tick)] - error_set[int(i/tick)]
        growth_set = np.append(growth_set, growth)

    plt.plot(t_tick, growth_set)
    plt.title('F={} T={}, dt={} error={}'.format(F, t_end, dt, err))
    plt.tick_params(labelsize = 8)
    plt.xticks(np.arange(0, t_end+dt, tick))

    if args.Save:
        filename = "Error_difference.png"
        plt.savefig(filename)

    plt.show()
