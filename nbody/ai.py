import pickle 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def build_encoder( input_dims=[3,20], layer_sizes=[ [32,32,32], [32,32,32] ],
                    combined_layers = [128,64,32], dropout=0.25,  output_dim=4 ):

    assert(len(input_dims)==len(layer_sizes))
    inputs = []
    layerx = []

    for i in range(len(input_dims)):
        inputs.append( tf.keras.Input(shape=(input_dims[i],), name='input_{}'.format(i)) )
        layerx.append( layers.Dense(layer_sizes[i][0], activation='relu')(inputs[i]) )

        for j in range(1,len(layer_sizes[i])):
            
            layerx[i] = layers.Dense(layer_sizes[i][j], activation='relu')(layerx[i])

    # combine the models
    c = layers.Concatenate()(layerx)

    for i in range(0,len(combined_layers)):
        c = layers.Dense(combined_layers[i], activation='relu')(c)
        if i == 0: c = layers.Dropout(dropout)(c)
        
    output = layers.Dense(output_dim, name='encoder_output',activation='sigmoid')(c)

    return tf.keras.Model(inputs=inputs, outputs=output, name='encoder')

def build_cnn_encoder( input_dims=[3,20], layer_sizes=[ [16,16,16], [16,32,32] ],
                    combined_layers = [128,64,32], dropout=0.25,  output_dim=4 ):

    assert(len(input_dims)==len(layer_sizes))
    inputs = []
    layerx = []

    for i in range(len(input_dims)):
        inputs.append( tf.keras.Input(shape=(input_dims[i],), name='input_{}'.format(i)) )
        
        if i == 1:
            resize = layers.Reshape((input_dims[i],1))(inputs[i])
            layerx.append( layers.Conv1D(layer_sizes[i][0], 3, strides=1)(resize) )
        else:
            layerx.append( layers.Dense(layer_sizes[i][0], activation='relu')(inputs[i]) )


        for j in range(1,len(layer_sizes[i])):
            if i == 1:
                layerx[i] = layers.Conv1D(layer_sizes[i][j], 3, strides=1)(layerx[i])
                layerx[i] = layers.BatchNormalization(momentum=0.75)(layerx[i]) 
            else:
                layerx[i] = layers.Dense(layer_sizes[i][j], activation='relu')(layerx[i])

    layerx[1] = layers.Flatten()(layerx[1])

    # combine the models
    c = layers.Concatenate()(layerx)

    for i in range(0,len(combined_layers)):
        c = layers.Dense(combined_layers[i], activation='relu')(c)
        if i == 0: c = layers.Dropout(dropout)(c)
        
    output = layers.Dense(output_dim, name='encoder_output',activation='sigmoid')(c)

    return tf.keras.Model(inputs=inputs, outputs=output, name='encoder')


def load_data(fname='Xy30_16.pkl', noise=None):

    X,z = pickle.load(open(fname,'rb'))
    X = np.array(X)
    y = X[:,3:]   # P2 [day], M2 [earth], omega2 [rad], ecc2
    X = X[:,:3]   # M* [sun], P1 [day], M1 [earth]
    z = np.array(z) # O-C data

    # noise up data since these are measurements after all
    if noise:
        Xn = np.copy(X)
        yn = np.copy(y) 
        zn = np.copy(z)

        for i in range(noise):
            # biased variation of points close to 0 O-C
            zr = z + np.random.normal(0,0.1*np.abs(z),z.shape) 

            Xn = np.concatenate( (Xn,X) )
            yn = np.concatenate( (yn,y) )
            zn = np.concatenate( (zn,zr) )

        return Xn,yn,zn
    else:
        return X, y, z

def ml_estimate(m, x,y):
    # TODO 
    return {'m':1,'P':1,'e':0,'omega':0}