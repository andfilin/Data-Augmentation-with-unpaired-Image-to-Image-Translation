import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Lambda

####
# temporal-mapping part of fcsrn.
# removes 1 dimension by averaging columns.
####
# n_filters = #classes+blank = 21 (10 digits, 10 half-digits, blank)
####
def temporalMapper(input_layer, n_filters=21):
    initializer = tf.random_normal_initializer(0., 0.01)
    t = input_layer
    # convolution, normalization
    t = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding="same", kernel_initializer=initializer)(t)
    t = BatchNormalization()(t)
    # reduce height-dimension by averaging columns
    t = Lambda(
        lambda x: tf.reduce_mean(x, 1, keepdims=False)
    )(t)
    
    return t
