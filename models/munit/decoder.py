from blocks import c7s1_k, dk, Rk, uk

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import tensorflow as tf

###
# takes content-code and style-code as input.
# outputs image.
###
def model(shape_content, shape_style, output_channels=3, smallModel=False):
    t_content = Input(shape=shape_content)
    t_style = Input(shape=shape_style)
    
    # generate adain-params from stylecode
    n_adain_layers = 4 if not smallModel else 3
    adain_channels = 256 if not smallModel else 128
    
    param_channels = t_content.shape[-1]
    n_mlp_outputs = 2*n_adain_layers*param_channels # output values beta, gamma for every adain-layer
    adain_params = mlp(shape_style, n_mlp_outputs)(t_style)
    
    # decode image from contentcode
    t = t_content

    activation = "relu"
    pad_type = "reflect"
    init = "he_normal"

    # residual-adain layers
    for i in range(n_adain_layers):
        adain_index = 2*i
        t = Rk_adain(t, adain_channels, activation, pad_type, adain_params, adain_index, param_channels, ( "dec_resAdain_%d" % (i) ) )
    channels = adain_channels // 2    
    
    
    # u128
    t = uk(channels, activation, pad_type, init, "dec_u128")(t)
    channels = channels // 2
    # u64
    t = uk(channels, activation, pad_type, init, "dec_u64")(t)
    
    # c7s1-3
    t = c7s1_k(output_channels, norm=None, activation="tanh", pad_type=pad_type, weight_init=init, name="dec_conv")(t)
    
    result = Model(inputs=[t_content, t_style], outputs=t)
    return result

###
# Residual Block with adain-normalizationlayer
###
# adain_params: array of every value to be used as beta,gamma by every adain-layer
# adain_index: index of first value of adain_params this layer can use.
# param_channels: size of each adain-param
def Rk_adain(input_layer, k, activation, pad_type, adain_params, adain_index, param_channels, name=None):
    norm = "none"
    init = "he_normal"
    
    gamma = tf.slice(adain_params, (0,adain_index*param_channels), (-1,param_channels))
    adain_index += 1
    beta = tf.slice(adain_params, (0,adain_index*param_channels), (-1,param_channels))
    
    gamma = tf.reshape(gamma, shape=(-1,1,1,param_channels))
    beta = tf.reshape(beta, shape=(-1,1,1,param_channels))
    
    t = input_layer
    t = Rk(t, k, norm, activation, pad_type, init, name)            
    t = adain_layer()( (t, gamma, beta) )
    
    return t

###
# adain-normalization layer:
# adain(tensor, gamma, beta) = gamma * ( (tensor - mean_channelwise(tensor) ) / std_channel(tensor) ) + beta
###
def adain(args):
    # unpack arguments
    input_tensor, gamma, beta = args
    t = input_tensor
    
    # mean, std
    axis = [1,2]
    shape = [-1, t.shape[1], t.shape[2], t.shape[3]]
    #channels = t.shape[3]
    
    mean = tf.math.reduce_mean(t, axis=axis, keepdims=True)
    
    #mean = tf.reshape(mean, shape=shape)
    std = tf.math.reduce_std(t, axis=axis, keepdims=True)
    #std = tf.reshape(std, shape=shape)
    
    result = (t - mean) / std
    #import pdb; pdb.set_trace()
    result = gamma * result + beta
    
    return result 

###
# lambda-layer of adain
###
def adain_layer():
    return tf.keras.layers.Lambda(
        adain
    )
    
        

    
###
# MultiLayerPerceptron generating parameters for adain-layers
###
# input_shape: shape of style-code: should be [n,8]
# output_channels: How many values to output.
#                  -> (Beta, Gamma) for each adain-layer -> 2*n_adain-layers*content_code_channels
def mlp(input_shape, output_channels, mlp_channels=256, n_blk=3):
   # outputchannel == number of needed parameters
    t_input = Input(shape=input_shape)
    t = t_input
    
    for i in range(n_blk - 1):
        t = Dense(mlp_channels, activation="relu", use_bias=True, name=("mlp_%d"%(i)) )(t)
    t = Dense(output_channels, activation=None, use_bias=True, name=("mlp_%d" % (n_blk-1)) )(t)
    
    result = Model(inputs=t_input, outputs=t)
    return result


    
    
