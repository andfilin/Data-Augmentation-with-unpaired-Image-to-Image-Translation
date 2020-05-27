from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.layers import Input, Conv2D
from generator import generator as generator_modified
from generator import generator_sizeDependant
from discriminator import discriminator as discriminator_modified
from discriminator import discriminator_sizeDependant

####
# returns a generator.
####
def generator(image_shape, norm_type='instancenorm'):
    #return pix2pix.unet_generator(output_channels, norm_type)
    return generator_modified(image_shape)
    #return generator_sizeDependant(image_shape)
####
# returns a discriminator.
####
def discriminator(n_channels, norm_type='instancenorm'):
    #return pix2pix.discriminator(norm_type, target=False)
    return discriminator_modified(n_channels, norm_type)
    #return discriminator_sizeDependant(n_channels, norm_type)





























