from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D,LeakyReLU,Activation,Concatenate,BatchNormalization, Conv2DTranspose
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import load_model

from numpy.random import randint
from random import random
from numpy import ones, zeros
import numpy as np
import time
from pathlib import Path
from datetime import datetime


"""
CycleGAN as class.

based on implementetation from:
https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/

No further changes for the task of meterValueTranslation were made.

Usage:
    1. create cyclegan-instance by supplying list: (imagesDomainA, imagesDomainB) as dataset.
        > cgan = cyclegan(dataset, savepath)
    2. call function 'startTraining(n_epochs)' to start training
        > cgan.startTraining(epochs=5, batchsize=1)
    3.a to translate images retrieve generator from object,
        > gen = cgan.gen_AtoB
    3.b save Models with function 'saveModels()'
        > cgan.saveModels()
"""
class cyclegan:
    
    """
    arguments:
        dataset:
            list of 2 imagesets (domainA and domainB)
        savepath:
            pathlib.path folder where to save/load/log model
        testimages:
            images to generate testresults after each trainingsession
        n_resnet: 
            Number of resnet-blocks to use in generators
    """
    def __init__(self, dataset, savepath, testImages=None, n_resnet = 9):
        image_shape = dataset[0].shape[1:]
        self.dataset = dataset
        self.savepath = savepath
        self.testImages = testImages
        self.logfile = savepath / "log.txt"
        self.resultImages = savepath / "resultImages"
        self.totalTrainingsessions = 0
        self.totalEpochs = 0
        # if folder already exist, load model from there
        # else create folder and save new model.
        if savepath.exists():
            print("loading model from existing folder")
            self.loadModels()
            self.appendToLog("model loaded")
        else:
            # create Folder
            print("creating new model and folder")
            savepath.mkdir()
            # init model
            self.disc_A = self.define_disc(image_shape)
            self.disc_B = self.define_disc(image_shape)
            self.gen_AtoB = self.define_gen(image_shape, n_resnet)
            self.gen_BtoA = self.define_gen(image_shape, n_resnet)
            self.composite_forward = self.define_composite_model(self.gen_AtoB, self.disc_B, self.gen_BtoA, image_shape)
            self.composite_backward = self.define_composite_model(self.gen_BtoA, self.disc_A, self.gen_AtoB, image_shape)
            # save model
            self.saveModels()
            # create subfolder for generated images
            self.resultImages.mkdir()
            # init logfile
            self.logfile.touch()
            self.appendToLog("model created")

            
    def appendToLog(self, text):
        with self.logfile.open("a") as logf:
            logf.write(str(datetime.now()))
            logf.write("\t\t" + text + "\n")
            

        
        
    #####################    
    # define discriminator model
    #####################
    def define_disc(self, image_shape):
        # init weights randomly
        init = RandomNormal(stddev=0.02)
        
        # ------------
        # Layers
        # l1 - l4: Conv2D mit stride => x(0.5, 0.5, 2)
        # l5: Conv2D ohne stride => x(1, 1, 1)
        # output: 1 Feature
        # ------------
        
        # input tensor
        in_image = Input(shape=image_shape)
        
        # L1: C64 == 64 Conv2D-Filters
        t = Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(in_image)
        t = LeakyReLU(alpha=0.2)(t)
        
        # L2: C128
        t = Conv2D(128, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(t)
        t = InstanceNormalization()(t)
        t = LeakyReLU(alpha=0.2)(t)
        
        # L3: C256
        t = Conv2D(256, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = LeakyReLU(alpha=0.2)(t)
        
        # L4: C512
        t = Conv2D(512, (4,4), strides=(2,2), padding="same", kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = LeakyReLU(alpha=0.2)(t)
        
        # L5: C512 ohne strides
        t = Conv2D(512, (4,4), padding="same", kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = LeakyReLU(alpha=0.2)(t)
        
        # patch output
        patch_out = Conv2D(1, (4,4), padding="same", kernel_initializer=init)(t)
        # ------------
        
        # define model
        model = Model(in_image, patch_out)
        model.compile(loss="mse", optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        return model 
    
    #####################
    # helperfunction for implementing resnet-blocks:
    # 2 conv2d-layers, where the input to the first layer is concatenated 
    # to the output of the seconde layer
    #####################
    def resnet_block(self, n_filters, input_layer):
        # random initial weights
        init = RandomNormal(stddev=0.02)
        # first layer
        t = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
        t = InstanceNormalization(axis=-1)(t)
        t = Activation('relu')(t)
        # second layer
        t = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        # concatenate input to output
        t = Concatenate()([t, input_layer])
        return t
    
    #####################
    # define generator model
    #####################
    def define_gen(self, image_shape, n_resnet):
        # random initial weights
        init = RandomNormal(stddev=0.02)
        
        # ------------
        # Layers
        # 3 Schritte:
        # I. Encodieren mit ConvLayers (Bild kleiner machen)
        #     LA1, LA2, LA3
        # II. Code verändern mit ResNetBlöcken
        # III. Decodieren mit ConvTransposeLayers (Bild aus Code machen)
        #     LB3, LB2, LB1
        #
        # => inputshape == outputshape !
        # ------------
        
        # input
        in_image = Input(shape=image_shape)
        
        # LA1 -> (x, y, 64)
        t = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
        t = InstanceNormalization(axis=-1)(t)
        t = Activation('relu')(t)
        
        # LA2 -> (0.5x, 0.5y, 128)
        t = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = Activation('relu')(t)
        
        # LA3 -> (0.5x, 0.5y, 256)
        t = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = Activation('relu')(t)
        
        # Residual Blocks
        for _ in range(n_resnet):
            t = self.resnet_block(256, t) # magic number?
            
        # LB3 -> (2x, 2y, 128)
        t = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = Activation('relu')(t)
        
        # LB2 -> (2x, 2y, 64)
        t = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        t = Activation('relu')(t)
        
        # LB1 -> (x, y, 3)
        t = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(t)
        t = InstanceNormalization(axis=-1)(t)
        out_image = Activation('tanh')(t)
        
        # ------------
        
        # define model
        model = Model(in_image, out_image)
        return model
    
    #####################
    # define composite model for training 1 Generator:
    # model Inputs: 
    #  input_gen from Domain 1,
    #  input_ident from Domain2
    # model outputs are 4 losses:
    #  Discriminator loss, identity loss, forward- and backward cycleloss
    #####################
    def define_composite_model(self, gen_1, disc, gen_2, image_shape):
        
        gen_1.trainable = True
        disc.trainable = False
        gen_2.trainable = False
        
        # ------------
        # discriminator elem
        # ------------
        input_gen = Input(shape=image_shape)
        gen1_out = gen_1(input_gen)
        disc_out = disc(gen1_out)
        # ------------
        # identity elem
        # ------------
        input_ident = Input(shape=image_shape)
        output_ident = gen_1(input_ident)
        # ------------
        # forward cycle
        # ------------
        cycle_forward_out = gen_2(gen1_out)
        # ------------
        # backward cycle
        # ------------
        gen2_out = gen_2(input_ident)
        cycle_backward_out = gen_1(gen2_out)
        
        # define model
        model = Model(
            # inputs
            [input_gen, input_ident], 
            # outputs
            [disc_out, output_ident, cycle_forward_out, cycle_backward_out]
                     )
        # optimizationAlgo
        opt = Adam(lr=0.0002, beta_1=0.5)
        # compile model
        #                          disc    ident   cycF  cycB                  
        model.compile(loss=[        "mse", "mae", "mae", "mae"],
                      loss_weights=[1,      5,     10,    10],
                     optimizer=opt)
        return model
    
    #####################
    # function to fetch trainingdata from some dataset
    #####################
    def fetch_samples(self, dataset, n_samples, patch_shape):
        randomIndices = randint(0, dataset.shape[0], n_samples)
        X = dataset[randomIndices]
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return X, y
    #####################
    # function to generate data for training
    #####################
    def generate_samples(self, generator, dataset, patch_shape):
        X = generator.predict(dataset)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y
    #####################
    # function for keeping a pool of recently generated images
    #####################
    def update_image_pool(self, pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                randomIndex = randint(0, len(pool))
                selected.append(pool[randomIndex])
                pool[randomIndex] = image
            else:
                selected.append(image)
        return np.asarray(selected)
    
    
    #####################
    # training function
    #####################
    def train(self, disc_A, disc_B, gen_AtoB, gen_BtoA, comp_forward, comp_backward, dataset, epochs, batchsize):
        #print("training for ", epochs, " epochs with a batchsize of " , batchsize)
        n_patch = disc_A.output_shape[1] # output square shape of discriminator
        trainA, trainB = dataset
        
        # prepare pools of generated images
        poolA, poolB = list(), list()
        # batches per epoch
        batchesPerEpoch = int(len(trainA) / batchsize) # for when batchsize != 1
        # total number of iterations
        n_steps = batchesPerEpoch * epochs
        
        # enumerate epochs
        for i in range(n_steps):
            starttime = time.time()
            ###
            # I. Fetch real images and generate artificial ones.
            ###
            # select a batch of real images for each domain
            X_realA, y_realA = self.fetch_samples(trainA, batchsize, n_patch)
            X_realB, y_realB = self.fetch_samples(trainB, batchsize, n_patch)
            # generate a batch of images for each domain
            X_genA, y_genA = self.generate_samples(gen_BtoA, X_realB, n_patch)
            X_genB, y_genB = self.generate_samples(gen_AtoB, X_realA, n_patch)
            # update pools of generated images
            X_genA = self.update_image_pool(poolA, X_genA)
            X_genB = self.update_image_pool(poolB, X_genB)
            #
            ###
            # II. use real and generated images to update every model
            ###
            # gen_BtoA
            gen_BtoA.trainable = True
            gen_AtoB.trainable = False
            disc_A.trainable = False
            loss_gen_BtoA, _, _, _, _ = comp_backward.train_on_batch(
                ## inputimages: 
                #sourceDomain, targetDomain
                [X_realB, X_realA],
                ## target values for:
                #DiscOut  Ident    cycleForward cycleBackward
                [y_realA, X_realA, X_realB,     X_realA]
                
            )
            # disc_A
            disc_A.trainable = True
            loss_discA_realInput = disc_A.train_on_batch(X_realA, y_realA)
            loss_discA_genInput = disc_A.train_on_batch(X_genA, y_genA)
            # gen_AtoB
            gen_AtoB.trainable = True
            gen_BtoA.trainable = False
            disc_B.trainable = False
            loss_gen_AtoB, _, _, _, _ = comp_forward.train_on_batch(
                [X_realA, X_realB],
                [y_realB, X_realB, X_realA, X_realB]
            )
            # disc_B
            disc_B.trainable = True
            loss_discB_realInput = disc_B.train_on_batch(X_realB, y_realB)
            loss_discB_genInput = disc_B.train_on_batch(X_genB, y_genB)
            ##
            # training done for this batch.
            ##
            ###
            # summarize performance
            print('>%d/%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1,n_steps,loss_discA_realInput,loss_discA_genInput,loss_discB_realInput,loss_discB_genInput,loss_gen_AtoB,loss_gen_BtoA))
            print("step took %f seconds" % (time.time() - starttime))
            
    #####################
    # function to use to start training.
    # trains on every image in dataset for n_epochs epochs
    #####################
    def startTraining(self, epochs, batchsize=1):
        print("training on %d images per domain for %d epochs and a batchsize of %d" % (self.dataset[0].shape[0], epochs, batchsize))
        
        imagesPerDomain = len(self.dataset[0])       
        self.appendToLog("starting training. Epochs = %d, imagesPerDomain = %d" % (epochs, imagesPerDomain) )
        
        starttime = time.time()
        self.train(self.disc_A, self.disc_B, self.gen_AtoB, self.gen_BtoA, self.composite_forward, self.composite_backward, self.dataset, epochs, batchsize)
        endtime = time.time() - starttime
        print("training finished after %f seconds" % (endtime))
        self.appendToLog("finished training after %f seconds" % (endtime))
        
        self.totalTrainingsessions += 1
        self.totalEpochs += epochs
        self.appendToLog("total: %d sessions and %d epochs completed." % (self.totalTrainingsessions, self.totalEpochs) )
    #####################
    # save every model
    # arg: dstPath - pathlib.Path object
    #####################
    def saveModels(self, dstPath=None):
        if(dstPath == None):
            dstPath = self.savepath
        
        # save gen_AtoB, compForward 
        self.gen_AtoB.trainable = True
        self.gen_BtoA.trainable = False
        self.disc_A.trainable = False
        self.disc_B.trainable = False
        self.gen_AtoB.save(dstPath / "gen_AtoB.h5")
        self.composite_forward.save(dstPath / "comp_forward.h5")
        # save gen_BtoA , compBackward
        self.gen_AtoB.trainable = False
        self.gen_BtoA.trainable = True
        self.disc_A.trainable = False
        self.disc_B.trainable = False
        self.gen_BtoA.save(dstPath / "gen_BtoA.h5")
        self.composite_backward.save(dstPath / "comp_backward.h5")
        # save disc_A
        self.disc_A.trainable = True        
        self.disc_A.save(dstPath / "disc_A.h5")
        # save disc_B
        self.disc_B.trainable = True
        self.disc_B.save(dstPath / "disc_B.h5")
        
        
        
    def loadModels(self, srcPath=None):
        if(srcPath == None):
            srcPath = self.savepath
        #import pdb; pdb.set_trace()
        self.gen_AtoB = load_model(str(srcPath / "gen_AtoB.h5"), compile=False)
        self.gen_BtoA = load_model(str(srcPath / "gen_BtoA.h5"), compile=False)
        self.disc_A = load_model(str(srcPath / "disc_A.h5"))
        self.disc_B = load_model(str(srcPath / "disc_B.h5"))
        self.composite_forward = load_model(str(srcPath / "comp_forward.h5"))
        self.composite_backward = load_model(str(srcPath / "comp_backward.h5"))
        
    """def gen_and_save_testimages(self):
        gen_images = self.gen_AtoB2.predict(self.testImages)"""
        
        