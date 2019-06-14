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

def ml_estimate(m, x,y):
    # TODO 
    return {'m':1,'P':1,'e':0,'omega':0}