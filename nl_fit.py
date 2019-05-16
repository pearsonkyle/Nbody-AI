from collections import OrderedDict
import argparse
import pprint
import copy 

import corner 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

from nbody.simulation import generate, analyze
from nbody.nested import lfit, nlfit, nbody_limits
from nbody.tools import msun, mearth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_ = "Input file with 3 columns of data (x,y,yerr)"
    parser.add_argument("-i", "--input", help=help_)

    help_ = "stellar mass"
    parser.add_argument("-ms", "--mstar", help=help_, default=1, type=float)

    help_ = "mid transit prior"
    parser.add_argument("-tm", "--tmid", help=help_, default=1, type=float)

    help_ = "planet 1 mass (earth)"
    parser.add_argument("-m1", "--mass1", help=help_, default=1, type=float)

    help_ = "planet 1 period (earth)"
    parser.add_argument("-p1", "--period1", help=help_, default=1, type=float)

    help_ = "planet 2 mass prior (earth)"
    parser.add_argument("-m2", "--mass2", help=help_, default=1, type=float)
    
    help_ = "planet 2 period prior (day)"
    parser.add_argument("-p2", "--period2", help=help_, default=1, type=float)

    help_ = "planet 2 omega prior (radian)"
    parser.add_argument("-o2", "--omega2", help=help_, default=0, type=float)

    help_ = "planet 2 eccentricity prior (radian)"
    parser.add_argument("-e2", "--eccentricity2", help=help_, default=0, type=float)

    help_ = "machine learning prior estimate"
    parser.add_argument("-ml", "--ai", help=help_, default=False, type=bool)

    args = parser.parse_args()

    data = np.loadtxt(args.input)

    # perform nested sampling linear fit to transit data    
    lstats, lposteriors = lfit( 
        data[:,0], data[:,1], data[:,2],
        bounds=[ 
            args.tmid-1, args.tmid+1, 
            args.period1-1, args.period1+1, 
        ] 
    )

    objects = [
        {'m':args.mstar},
        {'m':args.mass1*mearth/msun, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0 }, 
        {'m':args.mass2*mearth/msun, 'P':args.period2, 'e':args.eccentricity2, 'omega':args.omega2,  'inc':3.14159/2}
    ]

    # machine learning priors TODO 
    if args.ai:
        priors = ml_estimate(data[:,0], data[:,1])
        for k in priors.keys():
            objects[2][k] = priors[k]
    else:
        pass

    # estimate priors    
    bounds = [
        [ lstats['marginals'][0]['5sigma'][0], lstats['marginals'][0]['5sigma'][1] ], # mid transit (time)

        OrderedDict({
            'P':[lstats['marginals'][1]['5sigma'][0]-1./24, lstats['marginals'][1]['5sigma'][1]+1./24 ], # Period #1 (day)
        }),
        
        OrderedDict({
            'P':[ objects[2]['P']-1.5, objects[2]['P']+1.5 ],  # Period #2 (day)
            'm':[ objects[2]['m']-20*mearth/msun, objects[2]['m']+20*mearth/msun], # Mass #2 (msun)
            'e':[ max(objects[2]['e']-0.02,0), objects[2]['e']+0.02 ],
            'omega':[ objects[2]['omega']-1,  objects[2]['omega']+1 ],
        }),
    ]

    newobj, nlposteriors, nlstats = nlfit( data[:,0], data[:,1], data[:,2], objects, bounds )




    # Posterior #######################################################################################
    labels = [
        'Mid-transit [day]',
        'Period 1 [day]',
        'Period 2 [day]',
        'Mass 2 [Earth]',
        'Eccentricity 2',
        'Omega 2 [radian]',
    ]

    ranges = [ (nlstats['marginals'][i]['5sigma'][0], nlstats['marginals'][i]['5sigma'][1]) for i in range(len(nlstats['marginals'])) ]
    
    inf = cm.get_cmap('nipy_spectral', 256)
    newcmp = ListedColormap( inf(np.linspace(0,0.75,256))  )

    f = corner.corner(nlposteriors[:,2:], 
        labels= labels,
        bins=int(np.sqrt(nlposteriors.shape[0])), 
        range= ranges,
        plot_contours=False,
        plot_density=False,
        data_kwargs={'c':nlposteriors[:,1],'vmin':np.percentile(nlposteriors[:,1],1),'vmax':np.percentile(nlposteriors[:,1],50),'cmap':newcmp },
    )
    plt.savefig('ttv_posterior.png',bbox_inches='tight')
    plt.close()




    # O-C Model #######################################################################################
    ocdata_l = data[:,1] - ( data[:,0]*lstats['marginals'][1]['median'] + lstats['marginals'][0]['median'] )
    ocdata_nl = data[:,1] - ( data[:,0]*nlstats['marginals'][1]['median'] + nlstats['marginals'][0]['median'] )

    sim_data = generate(newobj, newobj[1]['P']* (data[:,0].max()+1), int( (data[:,0].max()+1)*newobj[1]['P']*24) )
    ttv_data = analyze(sim_data)
    upper, lower = nbody_limits( newobj, nlstats, ttv_data['planets'][0]['ttv'] )

    f,ax = plt.subplots(1, figsize=(7,4))
    ax.errorbar(data[:,0],ocdata_nl*24*60,yerr=data[:,2]*24*60,ls='none',marker='.',label='Data',color='black')
    ax.plot( ttv_data['planets'][0]['ttv']*24*60, label='Linear+Nbody ({:.1f})'.format(nlstats['global evidence']),color='red')
    ax.fill_between( np.arange(upper.shape[0]),24*60*upper,24*60*lower,alpha=0.1,label='Nbody 3 sigma',color='red')
    ax.axhline(ls='--',label='Linear ({:.1f})'.format(lstats['global evidence']))
    ax.legend(loc='best')
    ax.set_xlabel('Epochs')
    ax.set_xlim([min(data[:,0]),max(data[:,0])])
    ax.set_ylabel('O-C [min]')
    ax.grid(True)
    plt.savefig('ttv_model.png',bbox_inches='tight')
    plt.close()

    # TODO: test all command line arguments
    # run nl_fit.py -i sim_data.txt -p1 3.2888 -tm 0.82 -m1 79.45 -m2 31 -p2 7
    # 3.46 hours 