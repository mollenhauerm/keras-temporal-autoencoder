import keras
from keras.regularizers import l2,l1
import numpy as np

#-----------------------------------------------------------------------------

def Autoencoder(*size,activation='relu',optimizer='adadelta', loss='MSE'):
    
    input = keras.layers.Input(shape=(size[0],))
    encoded = input
    for s in size[1:]:
        encoded = keras.layers.Dense(s, activation=activation)(encoded)
    decoded = encoded
    for s in reversed(size[1:-1]):
        decoded = keras.layers.Dense(s, activation=activation)(decoded)
    decoded = keras.layers.Dense(size[0], activation='linear')(decoded)
    
    autoencoder = keras.models.Model(inputs=input, outputs=decoded)
    encoder = keras.models.Model(inputs=input, outputs=encoded)
    encoded_input = keras.layers.Input(shape=(size[-1],))
    decoder_layer = encoded_input
    for i in reversed(range(1, len(size))):
        decoder_layer = autoencoder.layers[-i](decoder_layer)
    
    decoder = keras.models.Model(inputs=encoded_input, outputs=decoder_layer,)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder, decoder

#-----------------------------------------------------------------------------

def L1Autoencoder(*size,activation='relu',optimizer='adadelta', loss='MSE',W_l1_penalty=0.01,b_l1_penalty=0.01):
    
    input = keras.layers.Input(shape=(size[0],))
    encoded = input
    for s in size[1:]:
        encoded = keras.layers.Dense(s, activation=activation,kernel_regularizer=l1(W_l1_penalty),
                                     bias_regularizer=l1(b_l1_penalty))(encoded)
    decoded = encoded
    for s in reversed(size[1:-1]):
        decoded = keras.layers.Dense(s, activation=activation,kernel_regularizer=l1(W_l1_penalty),
                                    bias_regularizer=l1(b_l1_penalty))(decoded)
    decoded = keras.layers.Dense(size[0], activation='linear',kernel_regularizer=l1(W_l1_penalty),
                                bias_regularizer=l1(b_l1_penalty))(decoded)
    
    autoencoder = keras.models.Model(inputs=input, outputs=decoded)
    encoder = keras.models.Model(inputs=input, outputs=encoded)
    encoded_input = keras.layers.Input(shape=(size[-1],))
    decoder_layer = encoded_input
    for i in reversed(range(1, len(size))):
        decoder_layer = autoencoder.layers[-i](decoder_layer)
    
    decoder = keras.models.Model(inputs=encoded_input, outputs=decoder_layer,)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder, decoder


#-----------------------------------------------------------------------------

def L2Autoencoder(*size,activation='relu',optimizer='adadelta', loss='MSE',W_l2_penalty=0.01,b_l2_penalty=0.01):
    
    input = keras.layers.Input(shape=(size[0],))
    encoded = input
    for s in size[1:]:
        encoded = keras.layers.Dense(s, activation=activation,kernel_regularizer=l2(W_l2_penalty),
                                     bias_regularizer=l2(b_l2_penalty))(encoded)
    decoded = encoded
    for s in reversed(size[1:-1]):
        decoded = keras.layers.Dense(s, activation=activation,kernel_regularizer=l2(W_l2_penalty),
                                    bias_regularizer=l2(b_l2_penalty))(decoded)
    decoded = keras.layers.Dense(size[0], activation='linear',kernel_regularizer=l2(W_l2_penalty),
                                bias_regularizer=l2(b_l2_penalty))(decoded)
    
    autoencoder = keras.models.Model(inputs=input, outputs=decoded)
    encoder = keras.models.Model(inputs=input, outputs=encoded)
    encoded_input = keras.layers.Input(shape=(size[-1],))
    decoder_layer = encoded_input
    for i in reversed(range(1, len(size))):
        decoder_layer = autoencoder.layers[-i](decoder_layer)
    
    decoder = keras.models.Model(inputs=encoded_input, outputs=decoder_layer,)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder, encoder, decoder
