from __future__ import absolute_import, division, print_function, unicode_literals

import pickle
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from nbody.tools import make_sin, lomb_scargle


def build_cnn_decoder(input_dim=7, layer_sizes=[32*2,32*4,32*8], conv_layers=[32,32,32,16], output_dim=32):

    inputs = tf.keras.Input(shape=(input_dim,), name='param_input')
    x = layers.Dense(layer_sizes[0], activation=tf.keras.layers.PReLU())(inputs)
    x = layers.Dropout(0.25)(x)
    #x = layers.GaussianNoise(0.1)(inputs)
    #gdec = tf.keras.Model(inputs=inputs, outputs=x, name='decoder')  

    for i in range(1,len(layer_sizes)):
        x = layers.Dense(layer_sizes[i], activation= tf.keras.layers.PReLU())(x)
    
    x = layers.Reshape((32,8))(x)

    for i in range(len(conv_layers)):
        x = layers.Conv1D(conv_layers[i], kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.75)(x)

    x = layers.Flatten()(x)

    output = layers.Dense(output_dim, name='decoder_output',activation='tanh') (x)

    return tf.keras.Model(inputs=inputs, outputs=output, name='decoder')

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

    z = np.array( [z[i][:32] for i in range(len(z))] )

    X, Xt, y, yt = train_test_split(X, z, test_size=0.1, random_state=42)
    Xpt = preprocessing.scale(Xt,axis=0)
    Xp = preprocessing.scale(X,axis=0)
    yp = y/y.max(0)
    yp[np.isnan(yp)] = 0

    decoder = build_cnn_decoder(
        input_dim=X.shape[1], 
        layer_sizes=[32*2, 32*4, 32*8], 
        conv_layers=[32, 32, 16], 
        output_dim=z.shape[1]
    )
    #decoder = build_decoder(X.shape[1], [512,256,256,128,64,32], y.shape[1])
    decoder.summary()
    
    try:
        decoder.load_weights("decoder.h5")
    except:
        print('load weights failed')
        pass 

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    #decoder.compile(optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError())
    decoder.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    decoder.fit(Xp, yp, epochs=args.epochs, batch_size=64)
    decoder.save_weights("decoder.h5")
    
    # Compute error in constructing O-C signals 
    Xv,yv,zv = pickle.load(open('X7_y20_z.pkl','rb'))
    zv = np.array( [zv[i][:32] for i in range(len(zv))] )

    Xp = preprocessing.scale(Xv,axis=0)

    ypred = decoder.predict(Xp)
    ypred *= y.max(0)

    error = np.abs(zv - ypred)
    
    plt.hist(error, bins=int(np.sqrt(error.shape[0])) )
    plt.xlabel("Average Error |OC - 3D Sin series| [min]")
    plt.show()
    plt.plot(zv[0],'k-'); plt.plot(ypred[0],'r-');plt.show()
    plt.scatter(Xv[:,4],Xv[:,3]/Xv[:,1],c=error.mean(1),vmin=0,vmax=2); plt.colorbar(); plt.show()
    


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
