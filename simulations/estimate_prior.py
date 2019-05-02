import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf

from nbody.ai import build_encoder 
from nbody.simulation import generate, analyze, report
from nbody.tools import mjup,msun,mearth

def load_data(fname='Xy30_6.pkl', npts=20, noise=False):

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
    # TODO 
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_, default="encoder.h5")
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=10 ,type=int)
    help_ = "Pickle file of training samples"
    parser.add_argument("-tr", "--train", help=help_)
    args = parser.parse_args()

    # train
    X,y,z = load_data(args.train, noise=True)
    Xs = X/X.max(0) # M* [sun], P1 [day], M1 [earth]
    zs = z/z.max(0) # O-C data 
    ys = y/y.max(0) # P2 [day], M2 [earth], omega2 [rad], ecc2

    encoder = build_encoder(
        input_dims=[X.shape[1],z.shape[1]], 
        layer_sizes=[ [8,8], [32,32] ],
        combined_layers = [128,64,32], 
        dropout=0.25,  
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


    ypred = encoder.predict([Xs,zs])
    ypred *= y.max(0)
    # P2, M2, omega2, ecc2 
    i = np.random.randint(X.shape[0])

    true_objects = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':y[i,1]*mearth/msun, 'P':y[i,0], 'inc':3.14159/2, 'e':y[i,3],  'omega':y[i,2]  }, 
    ]
    sim_data = generate(true_objects, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_true = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1])*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_pred = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]-0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper1 = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]+0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper2 = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0]+1, 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]-0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper11 = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]+19)*mearth/msun, 'P':ypred[i,0]-1, 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]+0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_upper22 = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]-9)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]-0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_lower1 = analyze(sim_data)

    pred = [
        {'m':X[i,0]},
        {'m':X[i,2]*mearth/msun, 'P':X[i,1], 'inc':3.14159/2, 'e':0, 'omega':0  }, 
        {'m':(ypred[i,1]-9)*mearth/msun, 'P':ypred[i,0], 'inc':3.14159/2, 'e':ypred[i,3],  'omega':ypred[i,2]+0.5  }, 
    ]
    sim_data = generate(pred, X[i,1]*20, int(X[i,1]*20*24) )
    ttv_lower2 = analyze(sim_data)


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

    plt.plot(ttv_true['planets'][0]['ttv']*24*60,'r-',label='Truth'.format(np.round(y[i],2)) )
    plt.errorbar(
        np.arange( ttv_true['planets'][0]['ttv'].shape[0] ),
        ttv_true['planets'][0]['ttv']*24*60+np.random.normal(0,0.25,ttv_true['planets'][0]['ttv'].shape[0]),
        yerr=np.random.uniform(0.25,0.5,ttv_true['planets'][0]['ttv'].shape[0]), ls='none',marker='o', label='Data', color='black'
    )
    plt.plot(ttv_pred['planets'][0]['ttv']*24*60,'g--', label='NN Estimate'.format(np.round(ypred[i],2)) )

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
