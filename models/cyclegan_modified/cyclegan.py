# add parent dir to syspath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.common_functions import plot_comparisonImage
from models.cyclegan_modified import submodels
import tensorflow as tf
import time
from IPython.core.debugger import set_trace

"""
cycleganmodel, based on:
https://www.tensorflow.org/tutorials/generative/cyclegan

modified to use a generator closer to the cyclegan-paper instead of the pix2pix-generator like in the example.

args:
    lambda: weight of cycleloss
    checkpoint_path: Path to folder where checkpoints are to be loaded from / saved to
    load_checkpoint_after_epoch: If given, loads a specific checkpoint from checkpoint_path. Else loads latest checkpoint
"""
class cyclegan():
    def __init__(self, image_shape, _lambda = 10, checkpoint_path = None, load_checkpoint_after_epoch=None):
        self._lambda = _lambda
        self.checkpoint_path = checkpoint_path
        n_channels = image_shape[2]
        # submodels
        self.gen_AtoB = submodels.generator(image_shape)
        self.gen_BtoA = submodels.generator(image_shape)
        self.disc_A = submodels.discriminator(n_channels)
        self.disc_B = submodels.discriminator(n_channels)
        # optimizers
        self.gen_AtoB_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.gen_BtoA_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_A_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_B_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        # prepare checkpoint
        if checkpoint_path != None:
            self.checkpoint = tf.train.Checkpoint(
                gen_AtoB = self.gen_AtoB,
                gen_BtoA = self.gen_BtoA,
                disc_A = self.disc_A,
                disc_B = self.disc_B,
                gen_AtoB_optimizer = self.gen_AtoB_optimizer,
                gen_BtoA_optimizer = self.gen_BtoA_optimizer,
                disc_A_optimizer = self.disc_A_optimizer,
                disc_B_optimizer = self.disc_B_optimizer                
            )
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=None, checkpoint_name="epoch")
            # load existing checkpoint if it exists
            if self.checkpoint_manager.latest_checkpoint:
                # load latest checkpoint if none specified
                if load_checkpoint_after_epoch == None:
                    checkpoint_to_be_loaded = self.checkpoint_manager.latest_checkpoint
                else:
                    checkpoint_to_be_loaded = str(checkpoint_path / ("epoch-%d" % (load_checkpoint_after_epoch)))
                # load checkpoint                                                                    
                self.checkpoint.restore(checkpoint_to_be_loaded)
                print("loaded checkpoint: ", checkpoint_to_be_loaded)
            else:
                # save inputshape in file
                shapeString = "%d,%d,%d" % (image_shape[0],image_shape[1],image_shape[2])
                shapeFile = checkpoint_path / "inputshape"
                shapeFile.write_text(shapeString)
                print("created new Model")
        else:
            self.checkpoint = None
            print("No checkpointpath given, model will not be saved.")
    
    ####
    # main training function
    ####
    # args:
    #    inputimages_A, inputimages_B: tf-Datasets with images to translate between
    #    testimages_A: tf-Dataset to create samples from, if != None
    #    n_testimages: number of images in testimages_A
    #    epochs: number of epochs to train
    #    epochs_before_save: After how many epochs checkpoint is to be saved, and a sample to be generated
    def train(self, inputimages_A, inputimages_B, testimages_A=None, n_testimages=0, epochs=4, epochs_before_save = 2):
        trainstart = time.time()
        for epoch in range(epochs + 1):
            epochstart = time.time()
            step = 0
            for image_A, image_B in tf.data.Dataset.zip((inputimages_A, inputimages_B)):
                stepstart = time.time()                
                self.train_step(image_A, image_B)
                print("step %d took: %f seconds" % (step, time.time() - stepstart))
                step += 1
            print("epoch %d took: %f seconds" % (epoch, time.time() - epochstart))
            # if checkpoint exists, save after every <epochs_before_save> epochs
            if self.checkpoint != None:
                totalEpochs = self.checkpoint.save_counter * epochs_before_save
                # every <epochs_before_save> epochs,                                                                   
                if (epoch % epochs_before_save) == 0:
                    # save checkpoint                                                               
                    savepath = self.checkpoint_manager.save(checkpoint_number=totalEpochs)
                    print("saved to: {}".format(savepath))
                    # additionaly create and save figure of samples if testimages are specified
                    if testimages_A != None:
                        figureSavepath = self.checkpoint_path / ("sample_epoch_%d.png" % (totalEpochs))
                        figWidth = 8 # in inches <-> 2.54cm
                        figHeight = 2 * n_testimages
                        plot_comparisonImage(self.gen_AtoB, testimages_A, figWidth, figHeight, figureSavepath)
            
        print("Training finished: %f seconds" % (time.time() - trainstart))
            
        
    ####    
    # for given discriminator outputs of real and generated images, 
    # calculates loss
    ####
    def _discriminator_loss(self, real, generated):
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_obj(tf.ones_like(real), real)
        generated_loss = loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
    ####
    # for given result of discriminator of generatoroutput, 
    # calculates loss
    ####
    def _generator_loss(self, generated):
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return loss_obj(tf.ones_like(generated), generated)
    ####
    # for given real image and result of cycling this image ( F(G(real)) ),
    # calculates cycleloss
    ####
    def _cycle_loss(self, real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self._lambda * loss
    ####
    # compares real image with the translated image to the same domain.
    ####
    def _identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self._lambda * 0.5 * loss
        
    @tf.function
    def train_step(self, real_A, real_B):
        with tf.GradientTape(persistent=True) as tape:
            # generator outputs: fake and cycled images
            gen_B = self.gen_AtoB(real_A, training=True)
            cycle_A = self.gen_BtoA(gen_B, training=True)
            gen_A = self.gen_BtoA(real_B, training=True)
            cycle_B = self.gen_AtoB(gen_A, training=True)
            # identity output
            ident_A = self.gen_BtoA(real_A, training=True)
            ident_B = self.gen_AtoB(real_B, training=True)
            # discriminator outputs
            disc_A_real = self.disc_A(real_A, training=True)
            disc_A_fake = self.disc_A(gen_A, training=True)
            disc_B_real = self.disc_B(real_B, training=True)
            disc_B_fake = self.disc_B(gen_B, training=True)
            
            # generator adversial losses
            gen_AtoB_loss = self._generator_loss(disc_B_fake)
            gen_BtoA_loss = self._generator_loss(disc_A_fake)
            # generators cycleloss
            total_cycle_loss = self._cycle_loss(real_A, cycle_A) + self._cycle_loss(real_B, cycle_B)
            
            # total losses
            total_gen_AtoB_loss = gen_AtoB_loss + total_cycle_loss + self._identity_loss(real_B, ident_B)
            total_gen_BtoA_loss = gen_BtoA_loss + total_cycle_loss + self._identity_loss(real_A, ident_A)
    
            disc_A_loss = self._discriminator_loss(disc_A_real, disc_A_fake)
            disc_B_loss = self._discriminator_loss(disc_B_real, disc_B_fake)
            
        ####
        # calculate gradients
        gen_AtoB_gradients = tape.gradient(total_gen_AtoB_loss, self.gen_AtoB.trainable_variables)
        gen_BtoA_gradients = tape.gradient(total_gen_BtoA_loss, self.gen_BtoA.trainable_variables)
        
        disc_A_gradients = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)
        disc_B_gradients = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)
        
        ####
        # apply gradients
        self.gen_AtoB_optimizer.apply_gradients(zip(gen_AtoB_gradients, self.gen_AtoB.trainable_variables))
        self.gen_BtoA_optimizer.apply_gradients(zip(gen_BtoA_gradients, self.gen_BtoA.trainable_variables))
        self.disc_A_optimizer.apply_gradients(zip(disc_A_gradients, self.disc_A.trainable_variables))  
        self.disc_B_optimizer.apply_gradients(zip(disc_B_gradients, self.disc_B.trainable_variables))
        
    ####
    # normalizes image to -1,1 and converts 3-channel.
    ####
    def preprocess_input(self, imageTensor):
        # reshape (h,w) -> (h,w,1)
        if len(imageTensor.shape) == 2:
            imageTensor = tf.reshape(imageTensor, (imageTensor.shape[0], imageTensor.shape[1], 1))
            # duplicate last dimension
            imageTensor = tf.repeat(imageTensor, 3, axis=-1) # should be the same as cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # to float
        imageTensor = tf.cast(imageTensor, tf.float32)
        # normalize
        imageTensor = (imageTensor / 127.5) - 1
        return imageTensor
        
   