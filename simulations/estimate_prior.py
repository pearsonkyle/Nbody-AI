import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf

from nbody.ai import build_encoder 
from nbody.simulation import generate, analyze, report, TTV
from nbody.tools import mjup,msun,mearth
 
from encoder_single import load_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    help_ = "Pickle file of training samples"
    parser.add_argument("-tr", "--train", help=help_)
    help_ = "Input file with 3 columns of data (x,y,yerr)"
    parser.add_argument("-i", "--input", help=help_)
    help_ = "stellar mass"
    parser.add_argument("-ms", "--mstar", help=help_, default=1, type=float)
    help_ = "planet 1 mass (earth)"
    parser.add_argument("-m1", "--mass1", help=help_, default=80, type=float)
    help_ = "planet 1 period (earth)"
    parser.add_argument("-p1", "--period1", help=help_, default=3.2888, type=float)
    args = parser.parse_args()
    parser = argparse.ArgumentParser()

    # train
    X,y,z = load_data(args.train, noise=10)
    Xs = X/X.max(0) # M* [sun], P1 [day], M1 [earth]
    zs = z/z.max(0) # O-C data [min]
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

    # quantify error in training data
    ypred = encoder.predict([Xs,zs])
    ypred *= y.max(0)
    res = yt-ypred


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


    ml_error = lambda x: x # TODO 

    nlstats = {
        'marginals':[
            {'median':0,'sigma':0}, # Tmid
            {'median':pred[1]['P'], 'sigma':0}, # P1
            {'median':pred[2]['P'], 'sigma': ml_error(pred[1]['P']) },       # P2
            {'median':pred[2]['m'], 'sigma': ml_error(pred[2]['m'])*mearth/msun }, # M2
            {'median':pred[2]['e'], 'sigma': ml_error(pred[2]['e']) },       # ecc2
            {'median':pred[2]['omega'],'sigma':ml_error(pred[2]['omega']) }  # w2
        ]
    }

    upper, lower = nbody_limits( pred, nlstatus, n=2)



    plt.errorbar(
        data[:,0],
        ttv*24*60,
        yerr=data[:,2]*24*60, 
        ls='none',marker='o', label='Data', color='black'
    )
    plt.plot(ttv_pred['planets'][0]['ttv']*24*60,'g--', label='NN Estimate')

    plt.fill_between(
        np.arange( ttv_true['planets'][0]['ttv'].shape[0] ),
        newmin*24*60,
        newmax*24*60,
        label='Prior Estimate',
        alpha=0.5,
        color='green'
    )

    plt.grid(True,ls='--')

    plt.ylabel('O-C Data [min]')
    plt.xlabel('Transit Epoch')
    plt.legend(loc='best')
    plt.show()

    # run estimate_prior.py -tr Xy30_10.pkl -i sim_data.txt -ms 1.12 -p1 3.2888 -m1 80 