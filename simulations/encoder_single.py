from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from nbody.ai import build_encoder 

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
    # TODO 
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_, default="encoder.h5")
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=10 ,type=int)
    help_ = "Pickle file of training samples"
    parser.add_argument("-tr", "--train", help=help_)
    help_ = "Pickle file of test samples"
    parser.add_argument("-te", "--test", help=help_)
    args = parser.parse_args()

    # train
    X,y,z = load_data(args.train, noise=True)
    Xs = X/X.max(0) # M* [sun], P1 [day], M1 [earth]
    zs = z/z.max(0) # O-C data 
    ys = y/y.max(0) # P2 [day], M2 [earth], omega2 [rad], ecc2

    # test
    Xt,yt,zt = load_data(args.test, noise=True)
    Xts = X/X.max(0)
    zts = z/z.max(0)
    yts = y/y.max(0)

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

    history = encoder.fit(
        [Xs,zs],
        ys,
        epochs=args.epochs, 
        batch_size=32,
        validation_data=([Xts,zts], yts),   
    )

    encoder.save_weights(args.weights)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    ypred = encoder.predict([Xts,zts])

    ypred *= y.max(0)
    res = yt-ypred

    f,ax = plt.subplots(2,2)
    ax[0,0].hist( res[:,0], bins=100, label=r"$\sigma$={:.2f}".format(np.std(res[:,0])))
    ax[0,0].set_xlabel('Period Error (day)')
    ax[0,0].legend(loc='best')
    ax[1,0].hist( res[:,1], bins=100, label=r"$\sigma$={:.2f}".format(np.std(res[:,1])))
    ax[1,0].set_xlabel('Mass Error (mearth)')
    ax[1,0].legend(loc='best')
    ax[0,1].hist( res[:,2], bins=100, label=r"$\sigma$={:.2f}".format(np.std(res[:,2])))
    ax[0,1].set_xlabel('omega Error')
    ax[0,1].legend(loc='best')
    ax[1,1].hist( res[:,3], bins=100, label=r"$\sigma$={:.2f}".format(np.std(res[:,3])))
    ax[1,1].set_xlabel('eccentricity Error')
    ax[1,1].legend(loc='best')
    plt.show()

    def add_colorbar(f,ax,im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical')

    f,ax = plt.subplots(2,2)
    im = ax[0,0].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,0]),s=2, vmin=0, vmax=3*np.abs(res[:,0]).std() )
    ax[0,0].set_xlabel('Period Error (day)')
    add_colorbar(f,ax[0,0],im)

    im = ax[1,0].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,1]),s=2, vmin=0, vmax=3*np.abs(res[:,1]).std() ) 
    ax[1,0].set_xlabel('Mass Error (mearth)')
    add_colorbar(f,ax[1,0],im)

    im = ax[0,1].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,2]),s=2, vmin=0, vmax=3*np.abs(res[:,2]).std() ) 
    ax[0,1].set_xlabel('omega Error')
    add_colorbar(f,ax[0,1],im)

    im = ax[1,1].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,3]),s=2, vmin=0, vmax=3*np.abs(res[:,3]).std() ) 
    ax[1,1].set_xlabel('eccentricity Error')
    add_colorbar(f,ax[1,1],im)

    plt.show()

    tf.keras.utils.plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=False)