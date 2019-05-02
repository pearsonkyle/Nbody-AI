from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import pywt
import tensorflow as tf
from tensorflow.keras import layers

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from nbody.tools import make_sin, lomb_scargle

def build_sin_decoder(input_dim=7, data_dim=50, layer_sizes=[128,32,32]):

    inputs = tf.keras.Input(shape=(input_dim,), name='param_input')

    x = layers.Dense(layer_sizes[0], activation='prelu')(inputs)
    ax = layers.Dropout(0.25)(x)
    wx = layers.Dropout(0.25)(x)
    px = layers.Dropout(0.25)(x)

    for i in range(1,len(layer_sizes)):
        ax = layers.Dense(layer_sizes[i], activation='prelu')(ax)
        ax = layers.BatchNormalization(momentum=0.75)(ax)

        wx = layers.Dense(layer_sizes[i], activation='prelu')(wx)
        wx = layers.BatchNormalization(momentum=0.75)(wx)

        px = layers.Dense(layer_sizes[i], activation='prelu')(px)
        px = layers.BatchNormalization(momentum=0.75)(px)

    out_a = layers.Dense(5, name='decoder_a',activation='sigmoid')(ax)
    out_w = layers.Dense(5, name='decoder_w',activation='sigmoid')(wx)
    out_p = layers.Dense(5, name='decoder_p',activation='sigmoid')(px)

    decoder = tf.keras.Model(inputs=inputs, outputs=[out_a,out_w,out_p], name='decoder')

    # Construct your custom loss as a tensor
    def custom_loss(y_true, y_pred):
        t = np.arange(data_dim)
        return  tf.keras.backend.sum( tf.keras.backend.sin( out_w*t + out_p)*out_a )

        #return tf.keras.losses.MeanAbsolutePercentageError(y_true, tf.keras.backend.sum( tf.keras.backend.sin( out_w*t + out_p)*out_a, axis=0) )


    # Add loss to model
    #decoder.add_loss(custom_loss)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    decoder.compile(optimizer, loss=custom_loss)

    return decoder


def build_decoder(input_dim=7, layer_sizes=[16,16,32,64], output_dim=24):

    inputs = tf.keras.Input(shape=(input_dim,), name='param_input')
    x = layers.Dense(layer_sizes[0], activation=tf.keras.layers.PReLU())(inputs)
    x = layers.Dropout(0.25)(x)
    #x = layers.GaussianNoise(0.1)(inputs)
    #gdec = tf.keras.Model(inputs=inputs, outputs=x, name='decoder')  

    for i in range(1,len(layer_sizes)):
        x = layers.Dense(layer_sizes[i], activation= tf.keras.layers.PReLU())(x)
        x = layers.BatchNormalization(momentum=0.75)(x)
    
    output = layers.Dense(output_dim, name='decoder_output',activation='linear') (x)

    return tf.keras.Model(inputs=inputs, outputs=output, name='decoder')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Number of training epochs"
    parser.add_argument("-e", "--epochs", help=help_, default=10 ,type=int)
    args = parser.parse_args()

    X,y,z = pickle.load(open('X7_y20_z.pkl','rb'))

    X, Xt, y, yt = train_test_split(X, y, test_size=0.1, random_state=42)
    Xpt = preprocessing.scale(Xt,axis=0)
    Xp = preprocessing.scale(X,axis=0)
    yp = y/y.max(0)
    yp[np.isnan(yp)] = 0

    decoder = build_decoder(X.shape[1], [64,64,128,128], y.shape[1])
    decoder.summary()
    
    try:
        decoder.load_weights("decoder.h5")
    except:
        print('load weights failed')
        pass 

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    decoder.compile(optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError())
    #decoder.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    #decoder.compile(optimizer, loss=custom_loss)
    decoder.fit(Xp, yp, epochs=args.epochs, batch_size=64)
    decoder.save_weights("decoder.h5")
    
    # Compute error in constructing O-C signals 
    Xv,yv,zv = pickle.load(open('X7_y20_z.pkl','rb'))

    Xp = preprocessing.scale(Xv,axis=0)

    ypred = decoder.predict(Xp)
    ypred *= y.max(0)

    error = np.zeros(Xv.shape[0])
    
    for i in range(Xv.shape[0]):

        Y = np.fft.fft( zv[i][:100] )

        py = np.copy(Y)
        py[:] = 0 
        n = 4
        freq = ypred[i][:n].astype(int)
        py[freq] = ypred[i][n:2*n] + 1j*ypred[i][3*n:4*n]
        py[-1*freq] = ypred[i][2*n:3*n] + 1j*ypred[i][4*n:5*n]

        ym = np.fft.ifft(py)

        res = z[i][:100]-ym
        error[i] = np.mean(np.abs(res))
    
    '''
    plt.hist(error, bins=int(np.sqrt(error.shape[0])) )
    plt.xlabel("Average Error |OC - 3D Sin series| [min]")
    plt.show()
    plt.plot(zv[i],'k-'); plt.plot(ym,'r-');plt.show()
    plt.scatter(Xv[:,4],Xv[:,3]/Xv[:,1],c=error); plt.colorbar(); plt.show()
    '''


    '''
    hey 
    what do you know
    up all night 
    I code like neo

    hey how do you want
    i code in the cloud
    and then you can clone it
    x2

    freestyle

    193 loud lord 
    '''
