from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from nbody.ai import build_encoder 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=10 ,type=int)
    args = parser.parse_args()

    f,ax = plt.subplots(2,2)

    for num in [10,20,30]:
        X,y,z = pickle.load(open('X7_y25_z.pkl','rb'))

        y = X[:,3:] # P2 [day], M2 [earth], omega2 [rad], ecc2
        X = X[:,:3] # M* [sun], P1 [day], M1 [earth]
        z = np.array( [z[i][:num] for i in range(len(z))] ) # O-C data
        
        # noise up data
        zn = z + np.random.normal(0,0.1*np.abs(z),z.shape)
        #zn = z + np.random.normal(0,0.5,z.shape)

        # scale features to ~0-1 
        ys = y/y.max(0)
        Xs = X/X.max(0)
        zs = z/z.max(0)

        # TODO update from old
        encoder = build_encoder(
            par_dim=X.shape[1], 
            oc_dim=z.shape[1],
            layer_sizes=[32*8,32*4,32*2], 
            conv_layers=[32,16,16], 
            output_dim=y.shape[1]
        )

        try:
            encoder.load_weights("encoder_{}.h5".format(num) )
        except:
            print('load weights failed')
            pass 

        encoder.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        encoder.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
        #encoder.compile(optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError())
        #encoder.compile(optimizer, loss=tf.keras.losses.MeanSquaredLogarithmicError())


        encoder.fit(
            { 'param_input':Xs, 'ocdata_input':zs.reshape(-1, zs.shape[1], 1) },
            ys,
            epochs=args.epochs, 
            batch_size=64
        )
        encoder.save_weights("encoder_{}.h5".format(num) )

        # open test data
        X,y,z = pickle.load(open('X7_y20_z.pkl','rb'))
        y = X[:,3:] # P2 [day], M2 [earth], omega2 [rad], ecc2
        X = X[:,:3] # M* [sun], P1 [day], M1 [earth]
        z = np.array( [z[i][:num] for i in range(len(z))] ) # O-C data
        zn = z + np.random.normal(0,0.1*np.abs(z),z.shape)

        # scale features to ~0-1 
        ys = y/y.max(0)
        Xs = X/X.max(0)
        zs = z/z.max(0)


        # quantify prediction error
        ypred = encoder.predict({
            'param_input':Xs,
            'ocdata_input':zs.reshape(-1, zs.shape[1], 1)
        })

        ypred *= y.max(0)
        res = y-ypred
        
        ax[0,0].hist( res[:,0], bins=100, alpha=0.5, label=r"{}: $\sigma$={:.2f}".format(num,np.std(res[:,0])) )
        ax[0,0].set_xlabel('Period Error (day)')
        ax[0,0].legend(loc='best')
        ax[1,0].hist( res[:,1], bins=100, alpha=0.5, label=r"{}: $\sigma$={:.2f}".format(num,np.std(res[:,1])))
        ax[1,0].set_xlabel('Mass Error (mearth)')
        ax[1,0].legend(loc='best')
        ax[0,1].hist( res[:,2], bins=100, alpha=0.5, label=r"{}: $\sigma$={:.2f}".format(num,np.std(res[:,2])))
        ax[0,1].set_xlabel('omega Error')
        ax[0,1].legend(loc='best')
        ax[1,1].hist( res[:,3], bins=100, alpha=0.5, label=r"{}: $\sigma$={:.2e}".format(num,np.std(res[:,3])))
        ax[1,1].set_xlabel('eccentricity Error')
        ax[1,1].legend(loc='best')

    plt.show()
