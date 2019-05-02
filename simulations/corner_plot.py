import pickle
import numpy as np 
import matplotlib.pyplot as plt
import argparse 
import corner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    help_ = "Input pickle file of simulations"
    parser.add_argument("-i", "--input", help=help_)
    args = parser.parse_args()


    X,z = pickle.load(open(args.input,'rb'))
    z = np.array( [z[i][:30] for i in range(len(z))] ) # O-C data

    keys = [
        'M*',
        'P1 [day]',
        'M1 [earth]',
        'P2 [day]',
        'M2 [earth]',
        'omega2 [rad]',
        'e2'
    ]
    corner.corner(X, labels=keys, quantiles=None, plot_contours=False, plot_density=False,
        data_kwargs={'c':np.max(z,1),'vmin':0,'vmax':10,'cmap':'jet'}, # custom library modification, switch plot->scatter
    )
    plt.show()