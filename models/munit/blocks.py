import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Layer, Add, UpSampling2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import InstanceNormalization

# ReflectionPadding-Implementation from:
# https://www.machinecurve.com/index.php/2020/02/10/using-constant-padding-reflection-padding-and-replication-padding-with-keras/
'''
  2D Reflection Padding
  Attributes:
    - padding: (padding_width, padding_height) tuple
'''
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')


def _convolution(n_filters, kernel_size, stride, norm, activation, pad_size, pad_type, weight_init, name=None):
    
    
    result = tf.keras.Sequential()
    
    """
    padding
    conv
    norm
    activation
    """
    
    # pad
    if pad_type == 'reflect':
        result.add(
            ReflectionPadding2D( padding=(pad_size, pad_size) )
        )
    elif pad_type == 'replicate':
        raise Exception("replication-padding not yet implemented")
    elif pad_type == 'zero':
        result.add(
            tf.keras.layers.ZeroPadding2D(padding=(pad_size, pad_size))
        )

    else:
        raise Exception("unknown padding-type: %s" % (pad_type) )
    
    
    # weigth init
    if weight_init == "normal":
        initializer = tf.random_normal_initializer(0., 0.02)
    elif weight_init == "he_normal":
        initializer = tf.keras.initializers.he_normal()
    else:
        raise Exception("unknown init-type: %s" % (weight_init) )
        
    # convolve
    result.add(
            Conv2D(n_filters, kernel_size, strides=stride, padding='valid', kernel_initializer=initializer, use_bias=False, name=name)
    )
    
    # normalize
    if norm == "instance":
        result.add(InstanceNormalization(axis=-1))
    elif norm == "layer":        
        print("!!! ------------------------")
        print("layer-normalization not implemented yet. for now, not using any normalization on upscaling layers. Implement later and compare results")
        print("!!! ------------------------")
       
        
    elif norm != None and norm != "none":
        raise Exception("unknown norm-type: %s" % (norm) )
        
    # activation
    if activation == "relu":
        result.add(tf.keras.layers.ReLU())
    elif activation == "lrelu":  
        result.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    elif activation == "tanh":
        result.add(tf.keras.layers.Activation('tanh'))
    elif activation != None and activation != "none":
        raise Exception("unknown activation-type: %s" % (activation) )

    return result
    
    
    
    
# convolution block
def c7s1_k(k, norm, activation, pad_type, weight_init, name=None):
    kernel_size = (7,7)
    stride = 1
    pad_size = 3    
    return _convolution(k, kernel_size, stride, norm, activation, pad_size, pad_type, weight_init, name)
    
# deconvolution block    
def dk(k, norm, activation, pad_type, weight_init, name=None):
    kernel_size = (4,4)
    stride = 2
    pad_size = 1
    return _convolution(k, kernel_size, stride, norm, activation, pad_size, pad_type, weight_init, name)

# upsampling block
def uk(k, activation, pad_type, weight_init, name=None):
    norm = "layer"
    result = tf.keras.Sequential()
    # 2x2 nearest neighbour upsampling
    result.add(
        UpSampling2D(size=(2, 2), interpolation='nearest')
    )
    # conv(k, (5,5), 1)
    result.add(
        _convolution(k, (5,5), 1, norm, activation=activation, pad_size=2, pad_type=pad_type, weight_init=weight_init, name=name)
    )
    
    return result
    
# residual block
def Rk(input_layer, k, norm, activation, pad_type, weight_init, name=None):
    if not name is None:
        name_res1 = name + "_A"
        name_res2 = name + "_B"
    else:
        name_res1 = None
        name_res2 = None
    kernel_size = (3,3)
    pad_size = 1
    stride=1
    t = input_layer
    t = _convolution(k, kernel_size, stride, norm, activation, pad_size, pad_type, weight_init, name_res1)(t)
    t = _convolution(k, kernel_size, stride, norm, "none", pad_size, pad_type, weight_init, name_res2)(t)
    output = Add()([t, input_layer])
    return output