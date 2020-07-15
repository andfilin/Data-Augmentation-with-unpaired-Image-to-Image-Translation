import encoder, decoder, discriminator

import tensorflow as tf
import time
import pandas as pd
from matplotlib import pyplot as plt
import io

class Munit():
    def __init__(self, image_shape, checkpoint_path = None, load_checkpoint_after_epoch=None, smallModel=False):
        
        self.image_shape = image_shape
        self.is_input_square = image_shape[0] == image_shape[1]
        
        self.checkpoint_path = checkpoint_path
        self.gan_loss = tf.keras.losses.MeanSquaredError()        
        self.lambda_image = 10
        self.lambda_style = 1
        self.lambda_content = 1
        self.lambda_cycle = 10
        
        output_channels = image_shape[-1]
        
        # Initialize Model (create generators, discriminators, optimizers)     
        self.init_models(image_shape, output_channels, smallModel=smallModel)
        
        # optimizers
        lr = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay
        self.init_optimizers(lr, lr_schedule)        
                
        # prepare checkpoint
        self.init_checkpoint(checkpoint_path, load_epoch=None)
        
        # lossmetrics
        self.init_lossMetrics()
    
    def init_models(self, image_shape, output_channels, smallModel):
        # encoders
        self.enc_content_A = encoder.content_model(image_shape, smallModel=smallModel)
        self.enc_style_A = encoder.style_model(image_shape, smallModel=smallModel)
        self.enc_content_B = encoder.content_model(image_shape, smallModel=smallModel)
        self.enc_style_B = encoder.style_model(image_shape, smallModel=smallModel)
        
        # shapes of codes
        self.content_shape = self.enc_content_A.output.shape[1:None] # shape exluding batchDim
        self.style_shape = self.enc_style_A.output.shape[1:None]
        output_channels = image_shape[-1] # outputchannels == inputchannels
        
        # decoders
        self.decoder_A = decoder.model(self.content_shape, self.style_shape, output_channels, smallModel=smallModel)
        self.decoder_B = decoder.model(self.content_shape, self.style_shape, output_channels, smallModel=smallModel)
        
        #disriminators
        #self.disc_A = discriminator.model(image_shape, smallModels=smallModels)
        #self.disc_B = discriminator.model(image_shape, smallModels=smallModels)
        self.disc_A = discriminator.multiscale_model(image_shape, smallModel=smallModel)
        self.disc_B = discriminator.multiscale_model(image_shape, smallModel=smallModel)
        
    def init_optimizers(self, lr, schedule=None):
        def optimizer(lr=lr, schedule=schedule):
            if not schedule is None:
                lr = schedule(lr,100000, decay_rate=0.5, staircase=True)
            return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
        
        self.opt_enc_c_A = optimizer()
        self.opt_enc_s_A = optimizer()
        self.opt_enc_c_B = optimizer()
        self.opt_enc_s_B = optimizer()
        self.opt_dec_A = optimizer()
        self.opt_dec_B = optimizer()
        self.opt_disc_A = optimizer()
        self.opt_disc_B = optimizer()
        
     
            
    def init_checkpoint(self, checkpoint_path, load_epoch=None):
        if checkpoint_path is None:
            self.checkpoint = None
            print("No checkpointpath given, model will not be saved.")
            return
        
        self.checkpoint = tf.train.Checkpoint(
                enc_content_A = self.enc_content_A,
                enc_style_A = self.enc_style_A,
                enc_content_B = self.enc_content_B,
                enc_style_B = self.enc_style_B,
                
                decoder_A = self.decoder_A,
                decoder_B = self.decoder_B,
                
                disc_A = self.disc_A,
                disc_B = self.disc_B,
                
                opt_enc_c_A = self.opt_enc_c_A,
                opt_enc_s_A = self.opt_enc_s_A,
                opt_enc_c_B = self.opt_enc_c_B,
                opt_enc_s_B = self.opt_enc_s_B,
                opt_dec_A = self.opt_dec_A,
                opt_dec_B = self.opt_dec_B,
                opt_disc_A = self.opt_disc_A,
                opt_disc_B = self.opt_disc_B,                                
            )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 
                                                                 checkpoint_path, 
                                                                 max_to_keep=None, 
                                                                 checkpoint_name="epoch")
        # check whether any checkpoint already exists
        if self.checkpoint_manager.latest_checkpoint:
            # load latest checkpoint if none specified
            if load_epoch == None:
                checkpoint_to_be_loaded = self.checkpoint_manager.latest_checkpoint
            else:
                checkpoint_to_be_loaded = str(checkpoint_path / ("epoch-%d" % (load_epoch)))
                self.checkpoint.restore(checkpoint_to_be_loaded)
                print("loaded checkpoint: ", checkpoint_to_be_loaded)
            return
            
        # no checkpoint exists,
        # save inputshape in file, save parameters in file
        shapeString = "%d,%d,%d" % (self.image_shape[0],self.image_shape[1],self.image_shape[2])
        shapeFile = checkpoint_path / "inputshape"
        shapeFile.write_text(shapeString)
        print("created new Model")
        # init summary
        summary_path = checkpoint_path / "logs"
        self.summary_writer = tf.summary.create_file_writer(str(summary_path))
        
        
    def init_lossMetrics(self):       
        self.lossName_generator = "loss_generator"
        self.lossName_gen_adversial = "loss_adversial_gen"
        self.lossName_gen_recon = "loss_recon"
        
        self.lossName_recon_image = "loss_recon_image"
        self.lossName_recon_style = "loss_recon_style"
        self.lossName_recon_content = "loss_recon_content"
        self.lossName_recon_cycle = "loss_recon_cycle"
        
        self.lossName_disc = "loss_discriminator"
        
        loss_names = [
            self.lossName_generator, self.lossName_gen_adversial, self.lossName_gen_recon,
            self.lossName_recon_image, self.lossName_recon_style, self.lossName_recon_content, self.lossName_recon_cycle,
            self.lossName_disc
        ]
        loss_accumulators = [tf.keras.metrics.Mean(name, dtype=tf.float32) for name in loss_names]
        self.dict_loss_accumulators = dict(zip(loss_names, loss_accumulators))
       
    ####
    # taking images from both domains,
    # calculates generator-loss.
    ####
    @tf.function
    def calc_gen_loss(self, input_A, input_B):
                                        
        ###
        # encode inputimages
        ###
        # A
        style_A = self.enc_style_A(input_A, training=True)
        content_A = self.enc_content_A(input_A, training=True)
        # B
        style_B = self.enc_style_B(input_B, training=True)
        content_B = self.enc_content_B(input_B, training=True)
                
        ###
        # decode encoded inputimages
        ###
        # A
        reconstructed_A = self.decoder_A([content_A, style_A], training=True)
        # A
        reconstructed_B = self.decoder_B([content_B, style_B], training=True)
        
        ###
        # sample random input styles
        ###       
        style_random_A = tf.random.normal(style_A.shape)
        style_random_B = tf.random.normal(style_B.shape) # style_A.shape == style_B.shape == (n,8)
        
        ###
        # decode crossdomain
        ###
        # content A, decoder B
        swapped_AB = self.decoder_B([content_A, style_random_B], training=True)
        # content B, decoder A
        swapped_BA = self.decoder_A([content_B, style_random_A], training=True)
        
        ###
        # encode crossed images
        ###
        # c A, s B
        content_AB = self.enc_content_B(swapped_AB, training=True)
        style_AB = self.enc_style_B(swapped_AB, training=True)
        # c B, s A
        content_BA = self.enc_content_A(swapped_BA, training=True)
        style_BA = self.enc_style_A(swapped_BA, training=True)
        
        ###
        # decode original image from decoded contents        
        ###
        cycled_A = self.decoder_A([content_AB, style_A], training=True)
        cycled_B = self.decoder_B([content_BA, style_B], training=True)
        
        ####
        # calculate reconstruction losses        
        ####        
        # image reconstruction loss
        loss_recon_image = self.reconstruction_loss( input_A, reconstructed_A ) * self.lambda_image
        loss_recon_image += self.reconstruction_loss( input_B, reconstructed_B ) * self.lambda_image
        
        # style reconstruction loss
        loss_recon_style = self.reconstruction_loss( style_A, style_BA) * self.lambda_style
        loss_recon_style += self.reconstruction_loss( style_B, style_AB) * self.lambda_style
        
        # content reconstructiong loss
        loss_recon_content = self.reconstruction_loss( content_A, content_AB) * self.lambda_content
        loss_recon_content += self.reconstruction_loss( content_B, content_BA) * self.lambda_content
        
        # cycled image reconstruction loss
        loss_recon_cycle = self.reconstruction_loss(input_A, cycled_A) * self.lambda_cycle
        loss_recon_cycle += self.reconstruction_loss(input_B, cycled_B) * self.lambda_cycle
        
        loss_recon = loss_recon_image + loss_recon_style + loss_recon_content + loss_recon_cycle
        
        ####
        # calculate Adversial Generator loss       
        ####
        loss_adversial_gen = self.adv_generator_loss(swapped_BA, self.disc_A) + self.adv_generator_loss(swapped_AB, self.disc_B)
        
        loss_total = loss_recon + loss_adversial_gen
        
        # log losses
        self.dict_loss_accumulators[self.lossName_generator](loss_total)
        self.dict_loss_accumulators[self.lossName_gen_adversial](loss_adversial_gen)
        self.dict_loss_accumulators[self.lossName_gen_recon](loss_recon)
        
        self.dict_loss_accumulators[self.lossName_recon_image](loss_recon_image)
        self.dict_loss_accumulators[self.lossName_recon_style](loss_recon_style)
        self.dict_loss_accumulators[self.lossName_recon_content](loss_recon_content)
        self.dict_loss_accumulators[self.lossName_recon_cycle](loss_recon_cycle)
        
        return loss_total
    
    ####
    # taking images from both domains,
    # calculates outputs needed for calculating discriminator-loss
    ####
    @tf.function
    def calc_disc_loss_inputs(self, input_A, input_B):
        ###
        # encode inputimages
        ###
        # A
        content_A = self.enc_content_A(input_A, training=False)
        # B
        content_B = self.enc_content_B(input_B, training=False)
        
        ###
        # sample random input styles
        ###
        shape = (input_A.shape[0], self.style_shape[0])
        style_random_A = tf.random.uniform(shape)
        style_random_B = tf.random.uniform(shape) # style_A.shape == style_B.shape == (n,8)
        
        ###
        # decode using inputcode+randomstyle, swapping contentcode
        ###
        # content A, Style B, decoder B
        swapped_AB = self.decoder_B([content_A, style_random_B], training=False)
        # content B, Style A, decoder A
        swapped_BA = self.decoder_A([content_B, style_random_A], training=False)
        return (swapped_AB, swapped_BA)
        #loss = discriminator.loss_disc(self.disc_A, input_A, swapped_BA, self.adversial_obj_disc)\
        #     + discriminator.loss_disc(self.disc_B, input_B, swapped_AB, self.adversial_obj_disc)
        #return loss
        
        
    ####
    # Losses
    ####
    
    # loss for decoding(encoding(something))
    @tf.function
    def reconstruction_loss(self, a, b):
        return tf.reduce_mean(tf.abs(a - b))
    
    # adversial loss of generator(enocoders+decoders) 
    @tf.function
    def adv_generator_loss(self, image_fake, disc):
        disc_out_fake = disc(image_fake, training=False)
        loss = self.gan_loss(disc_out_fake, 1.)
        return loss
    # adversial loss of discriminator
    @tf.function
    def adv_discriminator_loss(self, image_real, image_fake, disc):
        disc_out_fake = disc(image_fake, training=True)
        disc_out_real = disc(image_real, training=True)
        loss = self.gan_loss(disc_out_real, 1.) + self.gan_loss(disc_out_fake, 0.)
        return loss
    
    
    
    #def adversial_obj_gen(self, discOut_fake):
    #    return self.gan_loss(discOut_fake, 1.)
    #def adversial_obj_disc(self, discOut_real, discOut_fake):
    #    return self.gan_loss(discOut_real, 1.) + self.gan_loss(discOut_fake, 0.)
    @tf.function
    def train_step(self, input_A, input_B):
        ###
        # record gradients while calculating losses
        ###
        # generator-loss
        with tf.GradientTape(persistent=True) as tape_gen:
            loss_generator = self.calc_gen_loss(input_A, input_B)
            
        # discriminator-loss
        swapped_AB, swapped_BA = self.calc_disc_loss_inputs(input_A, input_B)
        with tf.GradientTape(persistent=True) as tape_disc:
            loss_discriminator = self.adv_discriminator_loss(input_A, swapped_BA, self.disc_A) + self.adv_discriminator_loss(input_B, swapped_AB, self.disc_B)
        # log mean of loss
        self.dict_loss_accumulators[self.lossName_disc](loss_discriminator)
        #self.loss_discriminator = loss_discriminator.numpy()
           
        # function applying gradients to a model given a tape, loss, model and optimizer
        def backprop(tape, loss, model, optimizer):
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            
        backprop_args = [
            # tape,    loss,           model,              optimizer
            (tape_gen, loss_generator, self.enc_content_A, self.opt_enc_c_A),
            (tape_gen, loss_generator, self.enc_style_A, self.opt_enc_s_A),
            (tape_gen, loss_generator, self.enc_content_B, self.opt_enc_c_B),
            (tape_gen, loss_generator, self.enc_style_B, self.opt_enc_s_B),
            
            (tape_gen, loss_generator, self.decoder_A, self.opt_dec_A),
            (tape_gen, loss_generator, self.decoder_B, self.opt_dec_B),
            
            (tape_disc, loss_discriminator, self.disc_A, self.opt_disc_A),
            (tape_disc, loss_discriminator, self.disc_B, self.opt_disc_B)
        ]       
        for tape, loss, model, opt in backprop_args:
            backprop(tape, loss, model, opt)                 
            
        
    def train(self, inputimages_A, inputimages_B, epochs=4, testimages_A=None, testimages_B=None, epochs_before_save=1, pauseBetweenEpochs=60):
        trainstart = time.time()
        for epoch in range(1, epochs + 1):
            print("epoch %d:" % (epoch) )
            epochstart = time.time()
            step = 0
            for image_A, image_B in tf.data.Dataset.zip((inputimages_A, inputimages_B)):
                stepstart = time.time()
                # trainstep
                losses = self.train_step(image_A, image_B)
                print("step %d took %.2f seconds" % (step, time.time()-stepstart) )
                step += 1 
            print("\nepoch %d took: %f seconds" % (epoch, time.time() - epochstart))
            self.sampleImages(testimages_A, testimages_B, epoch)
            if self.checkpoint != None:
                totalEpochs = (self.checkpoint.save_counter + 1) * epochs_before_save                
                # every <epochs_before_save> epochs,
                if (epoch % epochs_before_save) == 0:
                    # save checkpoint -                                                              
                    savepath = self.checkpoint_manager.save(checkpoint_number=totalEpochs)
                    self.log_losses(totalEpochs)
                    self.sampleImages(testimages_A, testimages_B, totalEpochs)
                    if self.is_input_square:
                        self.sampleImages(testimages_A, testimages_B, totalEpochs, stretch=4)
                    
            time.sleep(pauseBetweenEpochs)
        print("Training finished: %f seconds" % (time.time() - trainstart))
        
    def log_losses(self, epoch, printTime=False):
        if self.checkpoint_path is None:
            return
        starttime = time.time()
        with self.summary_writer.as_default():
            for name, loss in self.dict_loss_accumulators.items():
                tf.summary.scalar(name, loss.result(), step=epoch)
        print("wrote logs: %.2f seconds" % (time.time() - starttime) )

        """
        
        
        
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
            "loss_gen_total",
            "loss_recon_image", "loss_recon_style", "loss_recon_content", "loss_recon_cycle",
            "loss_gen_adv",
            "loss_disc_adv"
        ]
        list_losses = [
            self.loss_generator,
            self.loss_recon_image, self.loss_recon_style, self.loss_recon_content,
            self.loss_adversial_gen,
            self.loss_discriminator
        ]
        
        df = pd.DataFrame(list_losses, columns=labels)
        html_text = df.to_html()
        csv_text = df.to_csv(path_or_buf=csvpath)
        
        with htmlpath.open("a") as f:
            f.write("Epoch %d\n" % (epoch) )
            f.write(html_text)
        """
        
    def sampleImages(self, images_A, images_B, currentEpoch, show=True, save=True, stretch=0):
        # encode images
        content_A = self.enc_content_A(images_A)
        style_A = self.enc_style_A(images_A)
        content_B = self.enc_content_B(images_B)
        style_B = self.enc_style_B(images_B)
        
        # sample random styles
        style_random_A = tf.random.normal(style_A.shape)
        style_random_B = tf.random.normal(style_B.shape)
        
        # reconstruct original images
        rec_A = self.decoder_A([content_A, style_A])
        rec_B = self.decoder_B([content_B, style_B])
        
        # decode crossed images
        cA_sB = self.decoder_B([content_A, style_random_B])
        cB_sA = self.decoder_A([content_B, style_random_A])
        
        
        
        # stretch images
        if stretch > 0:
            new_dims = [ dim for dim in images_A.shape[1:3] ] # slice: [height, width]
            new_dims[1] *= stretch
            images_A = tf.image.resize(images_A, new_dims)
            images_B = tf.image.resize(images_B, new_dims)
            rec_A = tf.image.resize(rec_A, new_dims)
            rec_B = tf.image.resize(rec_B, new_dims)
            cA_sB = tf.image.resize(cA_sB, new_dims)
            cB_sA = tf.image.resize(cB_sA, new_dims)
            
            filename = ( "epoch_%d_stretch_%d.png" % (currentEpoch, stretch) )
        else:
            filename = ("epoch_%d.png" % (currentEpoch) )
            
        
        # generate and save figure
        sample_folder = self.checkpoint_path / "samples"
        if not sample_folder.exists():
            sample_folder.mkdir()
        sample_file = sample_folder / filename
        
        # figure contains rows each containing: 
        # inputA, inputB, recA, recB, ab, ba
        n_rows = images_A.shape[0]
        figDims = (32, n_rows)
        fig, a = plt.subplots(n_rows, 6, figsize=figDims, linewidth=1)
        for i in range(0,n_rows):
            # inputA
            a[i][0].imshow(images_A[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][0].axis("off")
            # inputB
            a[i][1].imshow(images_B[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][1].axis("off")
            
            # recA
            a[i][2].imshow(rec_A[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][2].axis("off")
            # recB
            a[i][3].imshow(rec_B[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][3].axis("off")
            
            # contentA-styleB
            a[i][4].imshow(cA_sB[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][4].axis("off")
            # contentB-styleA
            a[i][5].imshow(cB_sA[i,:,:,0], cmap="gray", vmin=-1,vmax=1)
            a[i][5].axis("off")
        a[0][0].set_title("input A")
        a[0][1].set_title("input B")
        a[0][2].set_title("reconstructed A")
        a[0][3].set_title("reconstructed B")
        a[0][4].set_title("contentA-styleB")
        a[0][5].set_title("contentB-styleA")
        
        fig.tight_layout(pad=0.5)
        if show:
            plt.show()
        if save:
            fig.savefig(sample_file)
            with self.summary_writer.as_default():
                tf.summary.image("Sample", self.plot_to_image(fig), step=currentEpoch)
        return None
    
    def plot_to_image(self, figure):        
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image