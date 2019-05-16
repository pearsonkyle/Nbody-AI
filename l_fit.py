import argparse 
import pprint
import corner 
import numpy as np
import matplotlib.pyplot as plt

from nbody.nested import lfit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_ = "Input file with 3 columns of data (x,y,yerr)"
    parser.add_argument("-i", "--input", help=help_)

    help_ = "slope prior"
    parser.add_argument("-m", "--slope", help=help_, default=1, type=float)

    help_ = "y-intercept prior"
    parser.add_argument("-b", "--yint", help=help_, default=1, type=float)

    args = parser.parse_args()

    data = np.loadtxt(args.input)

    # perform nested sampling linear fit to transit data    
    lstats, lposteriors = lfit( 
        data[:,0], data[:,1], data[:,2],
        bounds=[ 
            args.yint-1, args.yint+1, 
            args.slope-1, args.slope+1, 
        ] 
    )
    
    pprint.pprint(lstats)

    f = corner.corner(
        lposteriors[:,2:], 
        labels=['Y-intercept', 'Slope'],
        bins=int(np.sqrt(lposteriors.shape[0])), 
        range=[
            ( lstats['marginals'][0]['5sigma'][0], lstats['marginals'][0]['5sigma'][1]),
            ( lstats['marginals'][1]['5sigma'][0], lstats['marginals'][1]['5sigma'][1]),
        ],
        #no_fill_contours=True,
        plot_contours=False,
        plot_density=False,
        data_kwargs={'c':lposteriors[:,1],'vmin':np.percentile(lposteriors[:,1],10),'vmax':np.percentile(lposteriors[:,1],50) },
    )
    plt.show()
