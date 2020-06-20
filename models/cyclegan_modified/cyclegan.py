# add parent dir to syspath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from models.common_functions import plot_comparisonImage
import submodels
from ganMetrics.FID import FID_interface
from ganMetrics.GS import GS_interface
#from ganMetrics.save_samples import plot_samples

import tensorflow as tf
import time
import pandas as pd
import random
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.debugger import set_trace

from modelParameterNames import parameters as params

"""
cycleganmodel, based on:
https://www.tensorflow.org/tutorials/generative/cyclegan

modified to use a generator closer to the cyclegan-paper instead of the pix2pix-generator like in the example.

args:
    image_shape: shape of input/outputimages
    adversial_loss: lossfunction to use for Generator/Discriminator Adversial losses.
                    either "mse" (mean squared error) or "bce" (binary crossentropy)
    lr:  learning rate           
    lambda: weight of cycleloss
    checkpoint_path: Path to folder where checkpoints are to be loaded from / saved to
    load_checkpoint_after_epoch: If given, loads a specific checkpoint from checkpoint_path. Else loads latest checkpoint
    poolsize: how many fakeimages to buffer
"""
class cyclegan():
            
    def __init__(self, image_shape, n_images, batchsize, adversial_loss, lr=2e-4, _lambda = 10, checkpoint_path = None, load_checkpoint_after_epoch=None, poolsize=50):
        # store all parameters to log them later
        self.parameters = {
            params.model.value: self.__module__,            
            params.imageshape.value: image_shape,
            params.n_images.value: n_images,
            params._lambda.value: _lambda,
            params.lr.value: lr,
            params.adversial_lossfunction.value: adversial_loss,
            params.poolsize.value: poolsize,
            params.epochs_trained.value: 0,
            params.trainingsessions.value: 0,
            params.batchsize.value: batchsize,    
        }
        
        self.image_shape = image_shape
        self._lambda = _lambda
        self.checkpoint_path = checkpoint_path        
        if adversial_loss == "mse":
            self.adversial_loss = tf.keras.losses.MeanSquaredError()
        elif adversial_loss == "bce":
            self.adversial_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            raise Exception("Unknown adversial lossfunction: %s" % (adversial_loss) )
        
        
        
        n_channels = 1 if len(image_shape) == 2 else image_shape[2]
        
        ####
        # Initialize Model (create generators, discriminators, optimizers)
        ####        
        # submodels
        self.gen_AtoB = submodels.generator(image_shape)
        self.gen_BtoA = submodels.generator(image_shape)
        self.disc_A = submodels.discriminator(n_channels)
        self.disc_B = submodels.discriminator(n_channels)
        # optimizers
        self.gen_AtoB_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.gen_BtoA_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.disc_A_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        self.disc_B_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        
        # init pools of generated images
        self.poolsize = poolsize
        self.pool_A = []
        self.pool_B = []
        
        ####
        # prepare checkpoint
        ####
        if checkpoint_path != None:
            # init checkpoint
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
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 
                                                                 checkpoint_path, 
                                                                 max_to_keep=None, 
                                                                 checkpoint_name="epoch")
            # check whether any checkpoint already exists
            if self.checkpoint_manager.latest_checkpoint:
                # load latest checkpoint if none specified
                if load_checkpoint_after_epoch == None:
                    checkpoint_to_be_loaded = self.checkpoint_manager.latest_checkpoint
                else:
                    checkpoint_to_be_loaded = str(checkpoint_path / ("epoch-%d" % (load_checkpoint_after_epoch)))
                # load checkpoint; load parameters(for logging purposes)                                                                  
                self.checkpoint.restore(checkpoint_to_be_loaded)
                self.load_parameters()
                print("loaded checkpoint: ", checkpoint_to_be_loaded)
            else:
                # no checkpoint exists,
                # save inputshape in file, save parameters in file
                shapeString = "%d,%d,%d" % (image_shape[0],image_shape[1],image_shape[2])
                shapeFile = checkpoint_path / "inputshape"
                shapeFile.write_text(shapeString)
                
                self.log_parameters()
                print("created new Model")
        else:
            self.checkpoint = None
            print("No checkpointpath given, model will not be saved.")
            
            
    ####
    # inserts new_images into pool.
    # if full, randomly replace and return old image, or just return new_image    
    ####
    #@tf.function
    def update_pool(self, pool, new_images):
        if self.poolsize == 0:
            return new_images
        
        result = tf.TensorArray(tf.float32, dynamic_size=True, size=1)       
        resultIndex = 0        
        for image in new_images:            
            if len(pool) < self.poolsize:
                # pool not full
                pool.append(image)
                result = result.write(resultIndex,image); resultIndex += 1
            elif random.uniform(0,1) < 0.5:
                # pool full, replace random image
                index_replaced = random.randint(0,self.pool_size - 1) # stop is inclusive
                image_replaced = pool[index_replaced]
                pool[index_replaced] = image
                result = result.write(resultIndex,image_replaced); resultIndex += 1
            else:
                # pool full, return image without insertion
                result = result.write(resultIndex,image); resultIndex += 1

        result = result.stack()
        return result
    
    ####
    # main training function
    ####
    # args:
    #    inputimages_A, inputimages_B: tf-Datasets with images to translate between
    #    testimages_A: tf-Dataset to create samples from, if != None
    #    n_testimages: number of images in testimages_A
    #    epochs: number of epochs to train
    #    epochs_before_save: After how many epochs checkpoint is to be saved, and a sample to be generated
    #    loss_logsPerEpoch: how often to log losses every epoch
    def train(self, inputimages_A, inputimages_B, testimages_A=None, epochs=4, epochs_before_save = 1, metricsData=None, loss_logsPerEpoch=10, pauseBetweenEpochs=120):
        trainstart = time.time()
        totalsteps = tf.data.experimental.cardinality(inputimages_A).numpy()
        steps_before_log = int(totalsteps / loss_logsPerEpoch)
        steps_before_log = max(steps_before_log, 1) # in case loss_logsPerEpoch > totalsteps
        epochs_already_trained = self.checkpoint.save_counter * epochs_before_save
        # iterate epochs
        for epoch in range(1, epochs + 1):
            print("epoch %d:" % (epoch) )
            epochstart = time.time()
            losses_list = []
            # iterate steps            
            progBar = tf.keras.utils.Progbar(totalsteps)
            step = 0
            for image_A, image_B in tf.data.Dataset.zip((inputimages_A, inputimages_B)):
                stepstart = time.time()
                # trainstep
                losses = self.train_step(image_A, image_B)
                progBar.add(1)
                # log losses
                if (step % steps_before_log )== 0:
                    losses = [losstensor.numpy() for losstensor in losses]
                    losses_list.append(losses)                    
                step += 1                
             
            
            self.parameters[params.epochs_trained.value] += 1
            
            # log losses as html
            self.log_losses(losses_list, epoch + epochs_already_trained) 
           
            print("\nepoch %d took: %f seconds" % (epoch, time.time() - epochstart))
            # if checkpoint exists, save after every <epochs_before_save> epochs
            if self.checkpoint != None:
                totalEpochs = (self.checkpoint.save_counter + 1) * epochs_before_save                
                # every <epochs_before_save> epochs,
                if (epoch % epochs_before_save) == 0:
                    # save checkpoint -                                                              
                    savepath = self.checkpoint_manager.save(checkpoint_number=totalEpochs)
                    # - and save parameters (only number of trained epochs changed)
                    self.log_parameters()
                    print("saved to: {}".format(savepath))
                    
                    # additionaly create and save figure of samples if testimages are specified
                    if testimages_A != None:
                        self.save_sample(totalEpochs, testimages_A)
                        
            # sleep after every epoch to reduce strain and heat-builtup on gpu
            time.sleep(pauseBetweenEpochs)
            
        print("Training finished: %f seconds" % (time.time() - trainstart))
        self.parameters[params.trainingsessions.value] += 1
        self.log_parameters()
            
        
    ####    
    # for given discriminator outputs of real and generated images, 
    # calculates loss
    ####
    def _discriminator_loss(self, real, generated):
        #loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_obj = self.adversial_loss
        real_loss = loss_obj(tf.ones_like(real), real)
        generated_loss = loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5
    ####
    # for given result of discriminator of generatoroutput, 
    # calculates loss
    ####
    def _generator_loss(self, generated):
        #loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_obj = self.adversial_loss
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
            ####
            # generator outputs: fake and cycled images
            gen_B = self.gen_AtoB(real_A, training=True)
            cycle_A = self.gen_BtoA(gen_B, training=True)
            gen_A = self.gen_BtoA(real_B, training=True)
            cycle_B = self.gen_AtoB(gen_A, training=True)
            # pooled generated images for discriminatorlosses
            gen_B_pooled = self.update_pool(self.pool_B, gen_B)
            gen_A_pooled = self.update_pool(self.pool_A, gen_A)
            
            ####
            # generator identity output
            ident_A = self.gen_BtoA(real_A, training=True)
            ident_B = self.gen_AtoB(real_B, training=True)
            
            ####
            # discriminator outputs
            disc_A_real = self.disc_A(real_A, training=True)
            disc_A_fake = self.disc_A(gen_A, training=True)
            disc_B_real = self.disc_B(real_B, training=True)
            disc_B_fake = self.disc_B(gen_B, training=True)
            
            disc_A_fake_pooled = self.disc_A(gen_A_pooled, training=True)
            disc_B_fake_pooled = self.disc_B(gen_B_pooled, training=True)                        
            # generator adversial losses
            gen_AtoB_loss = self._generator_loss(disc_B_fake)
            gen_BtoA_loss = self._generator_loss(disc_A_fake)
            # generators cycleloss
            cycle_forward_loss = self._cycle_loss(real_A, cycle_A)
            cycle_backward_loss = self._cycle_loss(real_B, cycle_B)
            total_cycle_loss = cycle_forward_loss + cycle_backward_loss             
            # generator identity losses
            ident_B_loss = self._identity_loss(real_B, ident_B)
            ident_A_loss = self._identity_loss(real_A, ident_A)
            
            # total generator losses
            total_gen_AtoB_loss = gen_AtoB_loss + total_cycle_loss + ident_B_loss
            total_gen_BtoA_loss = gen_BtoA_loss + total_cycle_loss + ident_A_loss
    
            # discriminator losses
            disc_A_loss = self._discriminator_loss(disc_A_real, disc_A_fake_pooled)
            disc_B_loss = self._discriminator_loss(disc_B_real, disc_B_fake_pooled)
            
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
        # return losses
        losses = (
            disc_A_loss, disc_B_loss,
            total_gen_AtoB_loss, total_gen_BtoA_loss,
            gen_AtoB_loss, gen_BtoA_loss,
            cycle_forward_loss, cycle_backward_loss, total_cycle_loss,
            ident_B_loss, ident_A_loss
        )
        return losses
    
    ####
    # save every row of losses in list_losses in csv and html
    ####
    def log_losses(self, list_losses, epoch):
        if self.checkpoint_path is None:
            return
        # extra folder for losslogs
        lossFolder = self.checkpoint_path / "losses"
        if not lossFolder.exists():
            lossFolder.mkdir()
        # log losses of every epoch into one html-file
        htmlpath = lossFolder / "losses.html"       
        if not htmlpath.exists():
            htmlpath.touch()
        # log losses for every epoch into multiple csv-files
        csvpath = lossFolder / ( "epoch_%d.csv" % (epoch) )
        assert not csvpath.exists(), "Log for epoch %d should not exist already" % epoch 
        
        labels = [
            "disc_A_loss", "disc_B_loss",
            "total_gen_AtoB_loss", "total_gen_BtoA_loss",
            "gen_AtoB_loss", "gen_BtoA_loss",
            "cycle_forward_loss", "cycle_backward_loss", "total_cycle_loss",
            "ident_B_loss", "ident_A_loss"
        ]
        df = pd.DataFrame(list_losses, columns=labels)
        
        html_text = df.to_html()
        csv_text = df.to_csv(path_or_buf=csvpath)
        
        with htmlpath.open("a") as f:
            f.write("Epoch %d\n" % (epoch) )
            f.write(html_text)
    
    ####
    # write each parameter from map self.parameters to a file.
    ####
    def log_parameters(self):
        if self.checkpoint_path is None:
            return
        paramFile = self.checkpoint_path / "parameters.txt"
        if not paramFile.exists():
            paramFile.touch()
            
        text = ""
        for key, value in self.parameters.items():
            text += "%s:\t%s\n" % (key, value)
            
        paramFile.write_text(text)
    ####
    # load parameters from a file created by "log_parameters()" 
    # into dict "self.parameters"
    ####
    def load_parameters(self):
        assert not self.checkpoint_path is None, "tried to load parameters, but no path given"
        paramFile = self.checkpoint_path / "parameters.txt"
        if not paramFile.exists():
            print("No parameterfile in loaded Model")
            return
        paramsText = paramFile.read_text()
        
        # each line has form: "paramName:\tparamValue"
        lines = paramsText.split("\n")[0:-1] # slice out last line <== empty newline
        for param, value in [line.split(":\t") for line in lines]:
            self.parameters[param] = value
        
        # cast epochcount to int
        self.parameters[params.epochs_trained.value] = int(self.parameters[params.epochs_trained.value])
        self.parameters[params.trainingsessions.value] = int(self.parameters[params.trainingsessions.value])
        
        
        
        
    ####
    # Generate Images from images in metricsData[0];
    # for every row in metricsData[1] ([name, precalculated_stats]),
    # calculate FID to generated images and write to file.
    ####
    def caluclateMetrics(self, metricsData, metricsSavepath, totalEpochs):
        print("calculating FID...")
        generator_input, fid_stats = metricsData
        
        # predict, denormalize images for fid
        starttime = time.time()
        generated = self.gen_AtoB.predict(generator_input)        
        generated = self.denormalize_output(generated).astype("int")
        print("Generating %d images took %f seconds" % (len(generated), time.time() - starttime) )
        
        # calc score
        starttime = time.time()
        stats = FID_interface.calculate_stats(generated, printTime=False)
        
        if not metricsSavepath.exists():
            metricsSavepath.touch()
            metricsSavepath.write_text("FID for len(imageset) = %d\n\n" % (len(generated)) )
            
        fileText = "----------------------\n"
        fileText += "Epoch %d:\n" % (totalEpochs)
        for name, compareStats in fid_stats:
            fid = FID_interface.calculate_fid_from_stats(stats, compareStats)
            fileText += "fid(gen, %s) =\t%f\n" % (name, fid)
        print(fileText)
        #metricsSavepath.write_text(fileText)
        with metricsSavepath.open("a") as f:
            f.write(fileText)
        print("calculation of scores took %f seconds" % (time.time() - starttime) )
        
        
    ####
    # normalizes image to -1,1 and converts 3-channel.
    ####
    def preprocess_input(self, imageTensor):
        # reshape (h,w) -> (h,w,1)
        if len(imageTensor.shape) == 2:
            imageTensor = tf.reshape(imageTensor, (imageTensor.shape[0], imageTensor.shape[1], 1))
        #    # duplicate last dimension
        #    imageTensor = tf.repeat(imageTensor, 3, axis=-1) # should be the same as cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # to float
        imageTensor = tf.cast(imageTensor, tf.float32)
        # normalize
        imageTensor = (imageTensor / 127.5) - 1
        return imageTensor
        
    def denormalize_output(self, imageTensor):
        imageTensor = (imageTensor + 1) * 127.5
        return imageTensor
    
    ####
    # 
    ####
    def save_sample(self, epoch, testImageDataset, figDims=(16,20)):
        # prepare path
        sample_folder = self.checkpoint_path / "samples"
        if not sample_folder.exists():
            sample_folder.mkdir()
        sample_file = sample_folder / ("epoch_%d.png" % (epoch) )
        
        # 3 sets of images per sample
        translatedImages = self.gen_AtoB.predict(testImageDataset) # n,h,w,1; -1,1
        cycledImages = self.gen_BtoA.predict(translatedImages)
        inputImages = np.array( list(testImageDataset.as_numpy_iterator()) ) # n,b,h,w,1
        
        n_classes = inputImages.shape[0]
        assert n_classes == 20
        
        # make figure
        fig, a = plt.subplots(n_classes,3, figsize=figDims, linewidth=1)
        for i in range(0,n_classes):
            # first column: translated images
            #                               n,h,w,1
            a[i][0].imshow(translatedImages[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][0].axis("off")
            # second column: inputimages
            #                          n,b,h,w,1
            a[i][1].imshow(inputImages[i,0,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][1].axis("off")
            # third column: cycled images
            a[i][2].imshow(cycledImages[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][2].axis("off")
        a[0][0].set_title("translated")
        a[0][1].set_title("input")
        a[0][2].set_title("cycled")
        
        # save 
        fig.tight_layout(pad=0.5)
        fig.savefig(sample_file)
        
    