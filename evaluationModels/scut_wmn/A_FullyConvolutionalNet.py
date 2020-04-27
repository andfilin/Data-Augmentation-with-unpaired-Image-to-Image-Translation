import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input
####
# single convolution + Batchnorm (+ ReLU)
####
def convolution(n_filters, stride, activation=True):
    initializer = tf.random_normal_initializer(0., 0.01)
    result = tf.keras.Sequential()
    result.add(
        Conv2D(n_filters, kernel_size=(3,3), strides=stride, padding="same", kernel_initializer=initializer)
    )
    result.add(
        BatchNormalization()
    )
    if activation:
        result.add(
            Activation("relu")
        )
    return result
    
####
# 2 types of res-blocks
####
def resBlock_A(input_layer, n_filters):
    t = input_layer 
    t = convolution(n_filters, stride=1, activation=True)(t)
    t = convolution(n_filters, stride=1, activation=False)(t)
    t = Add()([t, input_layer])
    t = Activation("relu")(t)
    return t        
def resBlock_B(input_layer, n_filters):
    t = input_layer
    t = convolution(n_filters, stride=1, activation=True)(t)
    t = convolution(n_filters, stride=2, activation=False)(t)
    
    t2 = input_layer
    t2 = convolution(n_filters, stride=2, activation=False)(t2)
    
    output = Add()([t, t2])
    output = Activation("relu")(output)
    
    return output


####
# conolutional Part of fcsrn
####
def convNet(input_shape):
    inp = Input(input_shape)
    t = inp
    
    # first conv
    t = convolution(16, stride=1, activation=True)(t)
    # 3x resA(16)
    for _ in range(3):
        t = resBlock_A(t, 16)
    # resB(24)
    t = resBlock_B(t, 24)
    # 3x resA(24)
    for _ in range(3):
        t = resBlock_A(t, 24)
    # resB(32)
    t = resBlock_B(t, 32)
    # 3x resA(32)
    for _ in range(3):
        t = resBlock_A(t, 32)
    # resB(48)
    t = resBlock_B(t, 48)
    # 3x resA(48)
    for _ in range(3):
        t = resBlock_A(t, 48)
        
    output = t
    
    return (inp, output)
    