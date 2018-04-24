import numpy as np


class Lorenz96(object):
    def __init__(self, N_dim, F, init_x):
        self.F = F
        self.N = N_dim
        self.x = init_x
        self.t = 0


    def _lorenz(self, x, t):
        # Set initial dx
        dx = np.zeros(self.N)

        # Calculate exclusive dx[i] : dx[0], dx[1], dx[N-1]
        dx[0] = (x[1] - x[self.N-2]) * x[self.N-1] - x[0]
        dx[1] = (x[2] - x[self.N-1]) * x[0] - x[1]
        dx[self.N-1] = (x[0] - x[self.N-3]) * x[self.N-2] - x[self.N-1]

        # Calculate other dx
        for i in range(2, self.N-1):
            dx[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]

        dx += self.F

        return dx


    def Runge_Kutta_4(self, dt=0.01, t_end=30):
        # Make buffer of x, t
        t = self.t
        x = self.x
        x_buffer = np.array(np.array([x]))

        # Calculate Runge-Kutta method
        for _ in range(int(float(t_end)/dt)):
            q1 = self._lorenz(x, t)
            q2 = self._lorenz(x + (dt * q1)/2.0, t + dt/2.0)
            q3 = self._lorenz(x + (dt * q2)/2.0, t + dt/2.0)
            q4 = self._lorenz(x + (dt * q3), t + dt)
            k = dt * (q1 + q2 + q3 + q4) / 6.0
            x += k
            x_buffer = np.append(x_buffer, np.array([x]), axis=0)
            t += dt

        return x_buffer
