from blocks import dk

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D

def model(image_shape, smallModel=False):
    norm = "none"
    activation = "lrelu"
    pad_type = "reflect"
    init = "normal"
        
    in_image = Input(shape=image_shape)
    t = in_image
    
    # d64
    t = dk(64, norm, activation, pad_type, init, "d64")(t)
    # d128
    t = dk(128, norm, activation, pad_type, init, "d128")(t)
    # d256
    t = dk(256, norm, activation, pad_type, init, "d256")(t)
    
    if not smallModel:
        # d512
        t = dk(512, norm, activation, pad_type, init, "d512")(t)

    # output resultvalue
    t = Flatten()(t)
    t = Dense(1, activation=None, use_bias=True, kernel_initializer=init, name="disc_dense")(t)
    
    result = Model(inputs=in_image, outputs=t)
    return result

####
# get discriminator ouput once for a number of different scales.
####
def multiscale_model(image_shape, n_scales=3, smallModel=False):
    in_image = Input(shape=image_shape)
    t = in_image    
    
    # apply model once for each n_scales, downscaling after each time
    results = []
    result = model(image_shape)(t)
    results.append(result)
    for _ in range(n_scales - 1):
        t = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(t)
        result = model(t.shape[1:None], smallModel)(t)
        results.append(result)
       
    results = tf.reshape(results, (-1, n_scales) )
    
    return Model(inputs=in_image, outputs=results)

def loss_gen_multi(discriminator, input_fake, loss_obj):
    disc_outputs = discriminator(input_fake, training=True)
    loss_total = tf.Variable(0.)
    for disc_out in disc_outputs:
        loss = loss_obj(disc_out)       
        loss_total.assign_add(loss)
    return loss_total

def loss_gen_single(discriminator, input_fake, loss_obj):
    disc_outputs = discriminator(input_fake, training=True)
    return loss_obj(disc_outputs)

def loss_disc(discriminator, input_real, input_fake, loss_obj):
    outputs_real = discriminator(input_real, training=True)
    outputs_fake = discriminator(input_fake, training=True)
    loss_total = tf.Variable(0.)
    for out_real, out_fake in zip(outputs_real, outputs_fake):
        #loss = loss_obj(output_real, 1) + loss_obj(output_fake, 0)
        loss = loss_obj(out_real, out_fake)
        loss_total.assign_add(loss)
    return loss_total

def loss_disc_single(discriminator, input_real, input_fake, loss_obj):
    outputs_real = discriminator(input_real, training=True)
    outputs_fake = discriminator(input_fake, training=True)
    return loss_obj(outputs_real, outputs_fake)
    
    