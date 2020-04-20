from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.layers import Input, Conv2D
from models.cyclegan_modified.generator import generator as generator_modified
from models.cyclegan_modified.discriminator import discriminator as discriminator_modified

####
# returns a generator.
####
def generator(image_shape, norm_type='instancenorm'):
    #return pix2pix.unet_generator(output_channels, norm_type)
    return generator_modified(image_shape)
####
# returns a discriminator.
####
def discriminator(n_channels, norm_type='instancenorm'):
    #return pix2pix.discriminator(norm_type, target=False)
    return discriminator_modified(n_channels, norm_type)





























