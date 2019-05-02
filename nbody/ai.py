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

'''
def build_cnn_encoder(par_dim=3, oc_dim=20, layer_sizes=[32*8,32*4,32*2], conv_layers=[32,32,8], dropout=0.5, output_dim=4):

    params = tf.keras.Input(shape=(par_dim,), name='param_input')
    ocdata = tf.keras.Input(shape=(oc_dim,1), name='ocdata_input')

    x = layers.Conv1D(conv_layers[0], 3, strides=1)(ocdata)
    x = layers.BatchNormalization(momentum=0.75)(x)    

    for i in range(1,len(conv_layers)):
        x = layers.Conv1D(conv_layers[i], 3, strides=1)(x)
        #x = layers.BatchNormalization(momentum=0.75)(x)    

    x = layers.Concatenate()([params,x])
    x = layers.Dense(layer_sizes[0], activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    for i in range(1,len(layer_sizes)):
        x = layers.Dense(layer_sizes[i], activation='relu')(x)
        
    output = layers.Dense(output_dim, name='encoder_output',activation='sigmoid')(x)

    return tf.keras.Model(inputs=[params,ocdata], outputs=output, name='encoder')


def build_encoder(par_dim=3, oc_dim=25, layer_sizes=[32*8,32*4,32*2], dropout=0.25, output_dim=4):

    params = tf.keras.Input(shape=(par_dim,), name='param_input')
    ocdata = tf.keras.Input(shape=(oc_dim,1), name='ocdata_input')

    oc = layers.Flatten()(ocdata)
    x = layers.Concatenate()([params,oc])
    x = layers.Dense(layer_sizes[0], activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    for i in range(1,len(layer_sizes)):
        x = layers.Dense(layer_sizes[i], activation='relu')(x)
        
    output = layers.Dense(output_dim, name='encoder_output',activation='sigmoid')(x)

    return tf.keras.Model(inputs=[params,ocdata], outputs=output, name='encoder')
'''
