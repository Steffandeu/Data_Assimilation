import sys
import os
from itertools import count
import numpy as np

from lorenz96 import Lorenz96


date = 365 * 2
one_day = 0.2
one_year = 365 * one_day
total_t = int(date * one_day)
hour_6 = one_day/4

dt = 0.005
N_dim = 40
F = 8
t_end = 10
x = F * np.ones(N_dim)
x[N_dim-1] += 0.01

model = Lorenz96(N_dim=N_dim, F=F, init_x=x)

new_dir_path = "save_data"

if __name__ == '__main__':

    x = model.Runge_Kutta_4(dt=dt, t_end=date*one_day)

    # Latter 1 year
    x = x[int(one_year/dt+1):]

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for i in range(int(x.shape[0]/(hour_6/dt))):
        value = x[int(i*(hour_6/dt))]
        filename = new_dir_path+"/data"+str(i)+".dat"
        np.savetxt(filename, value)
