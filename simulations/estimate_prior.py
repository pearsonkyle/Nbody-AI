import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf

from nbody.ai import build_encoder 
from nbody.simulation import generate, analyze, report, TTV
from nbody.tools import mjup,msun,mearth

def load_data(fname='Xy30_6.pkl', npts=30, noise=False):

    X,z = pickle.load(open(fname,'rb'))

    y = X[:,3:]   # P2 [day], M2 [earth], omega2 [rad], ecc2
    X = X[:,:3]   # M* [sun], P1 [day], M1 [earth]
    z = np.array( [z[i][:npts] for i in range(len(z))] ) # O-C data
    
    # noise up data
    if noise:
        zn = z + np.random.normal(0,0.1*np.abs(z),z.shape)
        return X,y,zn

    else:
        return X, y, z

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    help_ = "Pickle file of training samples"
    parser.add_argument("-tr", "--train", help=help_)
    help_ = "Input file with 3 columns of data (x,y,yerr)"
    parser.add_argument("-i", "--input", help=help_)
    help_ = "stellar mass"
    parser.add_argument("-ms", "--mstar", help=help_, default=1, type=float)
    help_ = "planet 1 mass (earth)"
    parser.add_argument("-m1", "--mass1", help=help_, default=79.45, type=float)
    help_ = "planet 1 period (earth)"
    parser.add_argument("-p1", "--period1", help=help_, default=3.2888, type=float)
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    # train
    X,y,z = load_data(args.train, noise=True)
    Xs = X/X.max(0) # M* [sun], P1 [day], M1 [earth]
    zs = z/z.max(0) # O-C data 
    ys = y/y.max(0) # P2 [day], M2 [earth], omega2 [rad], ecc2

    data = np.loadtxt(args.input)
    ttv,p,tm = TTV(data[:,0], data[:,1])

    xf = np.array( [[args.mstar, args.period1, args.mass1]] )
    xf /= X.max(0)
    zf = np.array( [ttv*24*60] )
    zf /= z.max(0)

    encoder = build_encoder(
        input_dims=[X.shape[1],z.shape[1]], 
        layer_sizes=[ [8,8], [64,64] ],
        combined_layers = [128,128,32], 
        dropout=0.3,  
        output_dim=y.shape[1]
    )

    try:
        encoder.load_weights(args.weights)
    except:
        print('load weights failed')
        pass 

    encoder.summary()

    encoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    ypred = encoder.predict([xf,zf])
    ypred *= y.max(0)

    # P2, M2, omega2, ecc2 
    i = 0

    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1])*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]  }, 
    ]
    sim_data = generate(pred, max(data[:,1]), int(max(data[:,1])*24) ) 
    ttv_pred = analyze(sim_data)

    plt.errorbar(
        data[:,0],
        ttv*24*60,
        yerr=data[:,2]*24*60, 
        ls='none',marker='o', label='Data', color='black'
    )
    plt.plot(ttv_pred['planets'][0]['ttv']*24*60,'g--', label='NN Estimate')

    plt.show()

    # TODO fix this 

    '''
    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]-0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper1 = analyze(sim_data)

    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]+0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper2 = analyze(sim_data)

    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0]+1, 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]-0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper11 = analyze(sim_data)

    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0]-1, 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]+0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper22 = analyze(sim_data)

    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]-9)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]-0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_lower1 = analyze(sim_data)

    pred = [
        {'m':args.mstar},
        {'m':args.mass1, 'P':args.period1, 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]-9)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]+0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_lower2 = analyze(sim_data)
    '''

    newmin, newmax = [], []
    lists = [
        ttv_pred, ttv_upper1, ttv_upper2, ttv_lower1, ttv_lower2,
        ttv_upper11, ttv_upper22,
    ]
    for i in range(20):
        values = [ lists[j]['planets'][0]['ttv'][i] for j in range(len(lists))]
        newmin.append( np.min(values) )
        newmax.append( np.max(values) )
    newmin = np.array(newmin)
    newmax = np.array(newmax)

    # plot the results 
    #report(ttv_true, savefile='report.png')

    #plt.plot(ttv_true['planets'][0]['ttv']*24*60,'r-',label='Truth'.format(np.round(y[i],2)) )


    plt.fill_between(
        np.arange( ttv_true['planets'][0]['ttv'].shape[0] ),
        newmin*24*60,
        newmax*24*60,
        label='Prior Estimate',
        alpha=0.5,
        color='green'
    )

    plt.grid(True,ls='--')
    # TODO create a table of estimates for each parameter 
    plt.ylabel('O-C Data [min]')
    plt.xlabel('Transit Epoch')
    plt.legend(loc='best')
    plt.show()
