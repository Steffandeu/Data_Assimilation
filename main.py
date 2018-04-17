import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from lorenz96 import Lorenz96


# Set initial data for this model
N_dim = 30   # How many dimentions
F = 8        # Set the parameter of this model

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
    # Set initial parameter of x
    x = np.ones(N_dim)
    x[N_dim-1] += 0.01

    # Constructer
    model = Lorenz96(N_dim=N_dim, F=F, init_x=x)

    x = model.Runge_Kutta_4(dt=0.01, t_end=10)

    # plot first three variables
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.text2D(0.05, 0.95, "Lorez96", transform=ax.transAxes)
    ax.set_title('F={}'.format(F))
    ax.plot(x[:,0],x[:,1],x[:,3])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    if args.Save:
        filename = "F_"+str(F)+".png"
        plt.savefig(filename)

    plt.show()
