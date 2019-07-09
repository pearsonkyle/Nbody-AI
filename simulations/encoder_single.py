from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from nbody.ai import build_encoder, build_cnn_encoder, load_data

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
    X,y,z = load_data(args.train, noise=5)

    Xs = X/X.max(0) # M* [sun], P1 [day], M1 [earth]
    zs = z/z.max(0) # O-C data [min]
    ys = y/y.max(0) # P2 [day], M2 [earth], omega2 [rad], ecc2

    # test
    Xt,yt,zt = load_data(args.test, noise=2)
    Xts = Xt/X.max(0)
    zts = zt/z.max(0)
    yts = yt/y.max(0)

    dude()

    encoder = build_cnn_encoder(
        input_dims=[X.shape[1],z.shape[1]], 
        layer_sizes=[ [8,8,8], [16,32,32] ],
        combined_layers = [512,128,32], 
        dropout=0.5,  
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
        #loss=tf.keras.losses.MeanAbsolutePercentageError(),
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

    f,ax = plt.subplots(1)
    # Plot training & validation accuracy values
    #ax[0].plot(history.history['accuracy'])
    #ax[0].plot(history.history['val_accuracy'])
    #ax[0].set_ylabel('Accuracy')
    #x[0].set_xlabel('Training Epoch')
    #ax[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_ylabel('Loss')
    ax.set_xlabel('Training Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    #plt.show()
    plt.savefig('nn_training.pdf',bbox_inches='tight')
    plt.close()


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
    plt.savefig('nn_histogram.pdf',bbox_inches='tight')
    plt.close()

    def add_colorbar(f,ax,im,label):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = f.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label, rotation=270, labelpad=10)

    # create mask for all stars between 0.85-1.15 Msun
    f,ax = plt.subplots(2,2)
    im = ax[0,0].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,0]),s=2, vmin=0, vmax=3*np.abs(res[:,0]).std() )
    add_colorbar(f,ax[0,0],im,'Period Error [day]')
    ax[0,0].set_ylabel(r'Period Ratio (P$_{outer}$/P$_{inner}$)')

    im = ax[1,0].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,1]),s=2, vmin=0, vmax=3*np.abs(res[:,1]).std() ) 
    add_colorbar(f,ax[1,0],im,'Mass Error [Earth]')
    ax[1,0].set_ylabel(r'Period Ratio (P$_{outer}$/P$_{inner}$)')
    ax[1,0].set_xlabel('Mass of Outer Planet [Earth]')

    im = ax[0,1].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,2]),s=2, vmin=0, vmax=3*np.abs(res[:,2]).std() ) 
    add_colorbar(f,ax[0,1],im,'Arg of Periastron Error')

    im = ax[1,1].scatter(yt[:,1],yt[:,0]/Xt[:,1],c=np.abs(res[:,3]),s=2, vmin=0, vmax=3*np.abs(res[:,3]).std() ) 
    add_colorbar(f,ax[1,1],im,'Eccentricity Error')
    ax[1,1].set_xlabel('Mass of Outer Planet [Earth]')

    #plt.savefig('nn_error.pdf',bbox_inches='tight')
    #plt.close()
    plt.show()

    tf.keras.utils.plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=False)

    # TODO create model training plot
    # create model error plot
    # create mosaic of prior estimates 


    # redo abstract
    # read through 
    # machine learning figure errors 
    # table values for toi 193 
