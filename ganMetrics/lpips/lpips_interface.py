# add current dir to syspath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, current_dir)

from pathlib import Path
import pickle
import fid
from time import time
import numpy as np
import random
#import tensorflow.compat.v1 as tf_v1
#tf_v1.disable_v2_behavior()
import tensorflow as tf
import lpips_tf

#@tf_v1.function
def distance(image0, image1):
    #image0_ph = tf_v1.placeholder(tf_v1.float32)
    #image1_ph = tf_v1.placeholder(tf_v1.float32)
    distance_t  = lpips_tf.lpips(image0, image1, model='net-lin', net='alex')
    #with tf_v1.Session() as session:
    #    distance = session.run(distance_t)#, feed_dict={image0: image0, image1: image1})
    return distance_t

####
#
####
# i2imodel:      model whose diversity to test
# input_images:  input to model, normalized to [-1,1], shape [n,h,w,1]
# pairsPerInput: How many images to generate per inputImage
# splits:        number of fragments to split translated_images into; LPIPS is calculated seperately for each fragment (reduces required GPU-Memory)
# start_from_labels: whether the start of the generationchain are labels. If False, <input_images> will be used. Else generate inputimages from labels using synthGen
# fix_offsets: whether to have the same y_offsets of digits for the same inputlabel when generating images.
def mean_distance_from_i2imodel(i2imodel, input_images, n_inputs, pairsPerInput, splits=1, print_result=True, print_time=True, start_from_labels=False, synthGen=None, synthDims=None, fix_offsets=False):
    
    n_pairs = n_inputs * pairsPerInput
    
    
    if start_from_labels:
        # generate inputimages from <labels> using <synthGen>
        assert not synthGen is None and not synthDims is None
        # generate <n_inputs> labels
        labels = np.random.randint(0,20, (n_inputs, 5) )
        
        if fix_offsets:            
            # for each digit in labels, y_offset [0,1]
            normal_range = (-0.2,0.2)
            midstate_range=(0.3,0.7)
            offsets = [
                [
                     random.uniform(normal_range[0], normal_range[1]) if digit < 10
                     else random.uniform(midstate_range[0], midstate_range[1])
                for digit in label
                ] for label in labels
            ]
            
            offsets = np.repeat(offsets, pairsPerInput*2, axis=0)
        else:
            offsets = None
        
        
        # for each label, generate <pairsPerInput>*2 images -> repeat each label <pairsPerInput>*2 times
        labels = np.repeat(labels, pairsPerInput*2, axis=0)
        
        # generate images
        input_images = synthGen.makeImages(labels, resizeTo=synthDims, color=True, rotate=True, offsets=offsets)
        # prepare generated images for i2i-model: normalize to [-1,1], add channel-dim
        input_images = input_images.astype("float32")
        input_images = (input_images / 127.5) - 1
        # add channel-dims
        shape = input_images.shape
        if len(shape) < 4:
            shape = [d for d in shape]
            shape.append(1)
            input_images = np.reshape(input_images, shape)
    else:
        # use given inputimages
        assert len(input_images.shape) == 4 # (n, h, w, 1)
        assert input_images.shape[0] == n_inputs        
        # for each inputimage, generate <pairsPerInput>*2 translated images -> repeat each inputimage <pairsPerInput>*2 times
        input_images = np.repeat(input_images, pairsPerInput*2, axis=0)
     
    # make dataset of inputimages
    cgan_input = tf.data.Dataset.from_tensor_slices(input_images)\
        .cache()\
        .batch(1)
    
    # shape of result: [2*n_pairs,h,w,1]
    s_time = time()
    translated_images = i2imodel.gen_AtoB.predict(cgan_input)
    if print_time:
        print("translating %d images took %.2f seconds" % (n_pairs*2, time()-s_time) )
    
    # shuffle result
    tf.random.shuffle(translated_images)
    
    # reshape result into pairs: [n_pairs, 2, h, w, 1]
    shape = [d for d in translated_images.shape]
    shape[0] = int(shape[0]/2)
    shape.insert(1,2)
    translated_images = np.reshape(translated_images, shape)
    
    # split result into multiple fragments - to reduce required gpu-memory
    translated_images = tf.split(translated_images, splits)
    
    # calculate LPIPS for each fragment, keep sums
    s_time = time()
    sum_d = 0
    for set_of_pairs in translated_images:
        # LPIPS uses a pretrained model which only accepts 3-channel images
        images_a = tf.tile(set_of_pairs[:,0,:,:,:], [1, 1, 1, 3])
        images_b = tf.tile(set_of_pairs[:,1,:,:,:], [1, 1, 1, 3])
        
        d = distance(images_a,images_b)
        sum_d += tf.math.reduce_sum(d)
    # calculate mean distance from sum
    mean_d = sum_d / n_pairs
    
    if print_time:
        print("LPIPS took %.2f seconds" % (time()-s_time) )
        
    if print_result:
        print("mean(lpips): %f" % (mean_d) )
    
    return mean_d

