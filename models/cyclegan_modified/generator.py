import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation
from tensorflow.keras.initializers import RandomNormal
#from keras.models import Model
from tensorflow_addons.layers import InstanceNormalization

from resnet import resnet_block, ReflectionPadding2D


"""
From the cyclegan Paper:

Let c7s1-k denote a 7×7 Convolution-InstanceNorm-
ReLU layer with k filters and stride 1. dk denotes a 3 × 3
Convolution-InstanceNorm-ReLU layer with k filters and
stride 2. Reflection padding was used to reduce artifacts.
Rk denotes a residual block that contains two 3 × 3 con-
volutional layers with the same number of filters on both
layer. uk denotes a 3 × 3 fractional-strided-Convolution-
InstanceNorm-ReLU layer with k filters and stride 0.5.
The network with 6 residual blocks consists of:
c7s1-64,d128,d256,R256,R256,R256,
R256,R256,R256,u128,u64,c7s1-3

"""

def generator(image_shape, n_resBlocks=9, norm_type='instancenorm'):
    
    # number of resnet-blocks depends on imagesize, according to paper
    # 128x128: 6 blocks; 256x256: 9 blocks
    n_pixels = image_shape[0] * image_shape[1]
    if n_pixels >= (256*256):
        n_resBlocks = 9
    elif n_pixels >= (128*128):
        n_resBlocks = 6
    else:
        n_resBlocks = 3
    
    n_channels = image_shape[2]
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    in_image = Input(shape=image_shape)
    t = in_image
    # c7s1-64
    t = ReflectionPadding2D(padding=(3, 3))(t) 
    t = Conv2D(64, (7,7), padding="valid", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t)
    # d128 x,y,64
    t = Conv2D(128, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t)
    # d256 x,y,128
    t = Conv2D(256, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t) 
    
    # Resnet Blocks
    for _ in range(n_resBlocks):
        t = resnet_block(t, 256)
        
    # u128
    t = Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t) 
    # u64
    t = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t)
    # c7s1-3
    t = ReflectionPadding2D(padding=(3, 3))(t)
    t = Conv2D(n_channels, (7,7), padding="valid", kernel_initializer=init)(t)
   # t = InstanceNormalization(axis=-1)(t)
    output = Activation("tanh")(t)
    
    result = tf.keras.Model(inputs=in_image, outputs=output)
    return result

def generator_sizeDependant(image_shape, n_resBlocks=9, norm_type='instancenorm'):
    
    # number of resnet-blocks depends on imagesize, according to paper
    # 128x128: 6 blocks; 256x256: 9 blocks
    n_pixels = image_shape[0] * image_shape[1]
    if n_pixels >= (256*256):
        n_resBlocks = 9
    elif n_pixels >= (128*128):
        n_resBlocks = 6
    else:
        n_resBlocks = 3
    
    n_channels = image_shape[2]
    
    base_filterCount = 64 if n_channels == 3 else 32
    #base_filterCount =  int( 64 / ( (256*256) / n_pixels ) ) # if basefiltercount is 64 for 256x256 images, decrease it according to n_pixels
    
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    in_image = Input(shape=image_shape)
    t = in_image
    # c7s1-64
    t = ReflectionPadding2D(padding=(3, 3))(t) 
    t = Conv2D(base_filterCount, (7,7), padding="valid", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t)
    # d128 x,y,64
    t = Conv2D(base_filterCount*2, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t)
    # d256 x,y,128
    t = Conv2D(base_filterCount*4, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t) 
    
    # Resnet Blocks
    for _ in range(n_resBlocks):
        t = resnet_block(t, base_filterCount*4)
        
    # u128
    t = Conv2DTranspose(base_filterCount*2, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t) 
    # u64
    t = Conv2DTranspose(base_filterCount, (3,3), strides=(2,2), padding="same", kernel_initializer=init)(t)
    t = InstanceNormalization(axis=-1)(t)
    t = Activation("relu")(t)
    # c7s1-3
    t = ReflectionPadding2D(padding=(3, 3))(t)
    t = Conv2D(n_channels, (7,7), padding="valid", kernel_initializer=init)(t)
   # t = InstanceNormalization(axis=-1)(t)
    output = Activation("tanh")(t)
    
    result = tf.keras.Model(inputs=in_image, outputs=output)
    return result