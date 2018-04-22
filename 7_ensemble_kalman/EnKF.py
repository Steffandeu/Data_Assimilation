import numpy as np
from numpy.random import multivariate_normal


class EnKF(object):
    def __init__(self, x, P, dim_z, dt, N, hx, fx):
        self.dim_x = 40
        self.dim_z = 40
        self.dt = dt
        self.N = N
        self.hx = hx
        self.fx = fx

        self.Q = np.identity(self.dim_x)       # process uncertainty
        self.R = np.identity(self.dim_z)       # state uncertainty
        self.mean = np.sum(x)/self.dim_x
        self.initialize(x, P)

    def initialize(self, x, P):
        assert x.ndim == 1
        self.sigmas = multivariate_normal(mean=x, cov=P, size=self.N)

        self.x = x
        self.P = P


    def enkf(self, z, R=None):
        # predict
        N = self.N
        for i, s in enumerate(self.sigmas):
           self.sigmas[i] = self.fx(s, self.dt)

        e = multivariate_normal(self.mean, self.Q, N)
        self.sigmas += e
        #self.x = np.mean(self.sigmas , axis=0)

        P = 0
        for s in self.sigmas:
            sx = s - self.x
            P += np.outer(sx, sx)

        self.P = P / (N-1)


        # update
        if z is None:
            return

        if R is None:
            R = self.R
        if np.isscalar(R):
            R = np.identity(self.dim_z) * R

        N = self.N
        dim_z = len(z)
        sigmas_h = np.zeros((N, dim_z))

        # transform sigma points into measurement space
        for i in range(N):
            sigmas_h[i] = self.hx(self.sigmas[i])

        z_mean = np.mean(sigmas_h, axis=0)

        P_zz = 0
        for sigma in sigmas_h:
            s = sigma - z_mean
            P_zz += np.outer(s, s)
        P_zz = P_zz / (N-1) + R

        P_xz = 0
        for i in range(N):
            P_xz += np.outer(self.sigmas[i] - self.x, sigmas_h[i] - z_mean)
        P_xz /= N-1

        K = np.dot(P_xz, np.linalg.inv(P_zz))

        e_r = multivariate_normal([0]*dim_z, R, N)
        for i in range(N):
            self.sigmas[i] += np.dot(K, z + e_r[i] - sigmas_h[i])

        self.x = np.mean(self.sigmas, axis=0)
        self.P = self.P - np.dot(np.dot(K, P_zz), np.dot(K.T))
