import keras
from keras.regularizers import l2,l1
import numpy as np

#-----------------------------------------------------------------------------

def Autoencoder(*size,activation='relu',optimizer='adadelta', loss='MSE',regularization=None,W_penalty=.1,b_penalty=.1):
    
    input = keras.layers.Input(shape=(size[0],))
    encoded = input

    kernel_regularizer = None
    bias_regularizer = None

    if regularization == 'l1':
        kernel_regularizer = l1(W_penalty)
        bias_regularizer = l1(b_penalty)
    if regularization == 'l2':
        kernel_regularizer = l2(W_penalty)
        bias_regularizer = l2(b_penalty)

    for s in size[1:]:
        encoded = keras.layers.Dense(s, activation=activation,kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer)(encoded)
    decoded = encoded
    for s in reversed(size[1:-1]):
        decoded = keras.layers.Dense(s, activation=activation,kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=bias_regularizer)(decoded)
    decoded = keras.layers.Dense(size[0], activation='linear',kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer)(decoded)
    
    autoencoder = keras.models.Model(inputs=input, outputs=decoded)
    encoder = keras.models.Model(inputs=input, outputs=encoded)
    encoded_input = keras.layers.Input(shape=(size[-1],))
    decoder_layer = encoded_input
    for i in reversed(range(1, len(size))):
        decoder_layer = autoencoder.layers[-i](decoder_layer)
    
    decoder = keras.models.Model(inputs=encoded_input, outputs=decoder_layer,)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder, decoder
