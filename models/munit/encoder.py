from blocks import c7s1_k, dk, Rk

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def content_model(image_shape, smallModel=False):
    
    norm = "instance"
    activation = "relu"
    pad_type = "reflect"
    init = "he_normal"
    channels = 64 if not smallModel else 32
    n_resBlocks = 4 if not smallModel else 3
        
    in_image = Input(shape=image_shape)
    t = in_image
    
    # c7s1-64
    t = c7s1_k(channels, norm, activation, pad_type, init, "conv64")(t)
    channels *= 2
    
    # d128
    t = dk(channels, norm, activation, pad_type, init, "d128")(t)
    channels *= 2
    
    # d256
    t = dk(channels, norm, activation, pad_type, init, "d256")(t)
    
    # R256 x4
    for i in range(n_resBlocks):
        t = Rk(t, channels, norm, activation, pad_type, init, ( "res%d" % (i) ) )    
    
    result = Model(inputs=in_image, outputs=t)
    return result

def style_model(image_shape, smallModel=False):
    
    norm = "none"
    activation = "relu"
    pad_type = "reflect"
    init = "he_normal"
    style_dim = 8
    
    channels = 64 if not smallModel else 32
    n_downBlocks = 4 if not smallModel else 3
    
    in_image = Input(shape=image_shape)
    t = in_image
    
    #c7s1-64
    t = c7s1_k(channels, norm, activation, pad_type, init, "conv64")(t)
    channels *= 2
    
    # d128
    t = dk(channels, norm, activation, pad_type, init, "d128")(t)    
    channels *= 2
    for i in range(n_downBlocks - 1):
        # d256
        t = dk(channels, norm, activation, pad_type, init, ( "d256_%d" % (i) ) )(t)    
    
    # GAP - gloabal average pooling
    t = GlobalAveragePooling2D()(t)
    
    # fc8- fully connected layer
    init = tf.keras.initializers.he_normal()
    t = Dense(8, kernel_initializer=init, name="enc_style_dense")(t)
    
    result = Model(inputs=in_image, outputs=t)
    return result
    
    