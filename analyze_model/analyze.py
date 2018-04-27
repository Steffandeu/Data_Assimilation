import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lorenz96 import Lorenz96


class Analyze(object):
    def __init__(self, F):
        self.F = F
        self.N = 40
        self.dt = 0.01
        self.t_end = 4
        self.x_sigma = 1e-10
        self.noise_sigma = 1e-10
        self.sample_num = 15
        self.noise_sample_num = 32
        self.contour_frac = 3


    def initialize_x(self):
        # Initialize start point
        init_x = np.ones([self.N]) * self.F
        init_x += self.x_sigma * np.random.randn(self.N)

        return init_x


    def initialize_noise(self):
        # Initialize noise
        noise = self.noise_sigma * np.random.randn(self.N)

        return noise


    def error_develope(self):
        # sample various error development
        # Initialize initial x
        init_x = self.initialize_x() + self.initialize_noise()
        model = Lorenz96(N_dim=self.N, F=self.F, init_x=init_x)
        true_value = model.Runge_Kutta_4(dt=self.dt, t_end=self.t_end)

        # set buffer
        data_num = int(self.t_end/self.dt)+1
        buffer = np.zeros([data_num])

        # Sample noise included version
        for j in range(self.noise_sample_num):
            # Initialize noise
            x = init_x + self.initialize_noise()
            model = Lorenz96(N_dim=self.N, F=self.F, init_x=x)
            # Check error development
            value = model.Runge_Kutta_4(dt=self.dt, t_end=self.t_end)

            # Evaluate how development
            dev = value - true_value

            # Append to buffer
            for i in range(data_num):
                buffer[i] = np.linalg.norm(dev[i])

        return buffer


    def make_contour(self):
        t_num = int(self.t_end/self.dt + self.dt)
        t = np.arange(0, t_num, self.dt)
        init_x = self.initialize_x()
        model = Lorenz96(N_dim=self.N, F=self.F, init_x=init_x)
        value = model.Runge_Kutta_4(dt=self.dt, t_end=self.t_end)

        # Make hausdorff dimension by recursion system
        # x : x[t,N]
        x_dim = np.arange(0, self.N, float(1.0/(self.N ** (self.contour_frac-1))))
        x_num = int(self.N ** self.contour_frac)
        #x_buffer = np.tile(value.transpose(), x_num)

        # Plot
        plt.contour(value.transpose())
        plt.show()




    def show_image(self, x, save=False):
        # plot error growth
        t = np.arange(0, self.t_end+self.dt, self.dt)
        plt.plot(t, x[:len(t)])
        plt.title('F={} T={}, dt={}'.format(self.F, self.t_end, self.dt))
        plt.tick_params(labelsize = 8)
        #plt.xticks(np.arange(0, self.t_end+self.dt, 0.2))

        if save:
            plt.save(filename = "images/Lorenz_F_"+str(F)+".png")
            plt.savefig(filename)

        plt.show()


    def make_gif(self, x):
        fig = plt.figure()
        t = np.arange(0, self.t_end+self.dt, self.dt)

        ims = []

        ######## Process #######
        ########################
        ## Show example below ##
        #x = np.arange(0,10,0.1)
        #for a in range(50):
        #    y = np.sin(x - a)
        #    line, = plt.plot(x, y, "r")
        #    ims.append([line])
        ########################

        ani = animation.ArtistAnimation(fig, ims)
        ani.save('anim.gif', writer="imagemagick")
        ani.save('anim.mp4', writer="ffmpeg")
        plt.show()


if __name__ == "__main__":
    analyzer = Analyze(F=4)
    x = analyzer.error_develope()
    analyzer.make_contour()
    print(x.shape)
