# add root dir to syspath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parent_dir = os.path.dirname(current_dir)
#root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, current_dir)


from A_FullyConvolutionalNet import convNet
from B_TemporalMapper import temporalMapper

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode
import time

####
#  FULLY CONVOLUTIONAL SEQUENCE RECOGNITION NETWORK
####
# https://ieeexplore.ieee.org/document/8606091
####
class fcsrn():
    
    def __init__(self, input_shape, checkpoint_path = None, epoch_to_load=None):
        self.input_shape = input_shape
        self.checkpoint_path = checkpoint_path
        # build model
        input, convNet_output = convNet(input_shape)
        output = temporalMapper(convNet_output)
        output = Activation("softmax")(output)
        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0001)
        # prepare checkpoint
        if checkpoint_path != None:
            self.ar_file = checkpoint_path / "ar.csv"
            self.checkpoint = tf.train.Checkpoint(
                model = self.model,
                optimizer = self.optimizer,
            )
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=None, checkpoint_name="epoch")
            # load existing checkpoint if it exists
            if self.checkpoint_manager.latest_checkpoint:
                if epoch_to_load is None:                    
                    checkpoint_to_be_loaded = self.checkpoint_manager.latest_checkpoint
                else:
                    checkpoint_to_be_loaded = str(checkpoint_path / ("epoch-%d" % (epoch_to_load)))
                self.checkpoint.restore(checkpoint_to_be_loaded)
                print("loaded checkpoint: ", checkpoint_to_be_loaded)
            else:
                print("created new Model")
        else:
            self.checkpoint = None
            print("No checkpointpath given, model will not be saved.")
            
    def set_checkpointDir(self,newDir):
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, newDir, max_to_keep=None, checkpoint_name="epoch")
        self.ar_file = newDir / "ar.csv"
    ####
    # ctc-loss as lossfunction.
    # for now, without AugCtcLoss
    ####
    def lossfunction(self, modeloutput, target, batchsize, alpha_augLoss = 0):
        y_true = target
        y_pred = modeloutput
        n_time_slices = self.input_shape[1]/8 # fcsrn-model halves inputdimenions 3 times -> finalwidth = inputwidth / 8
        input_length = np.full((batchsize, 1), n_time_slices)
        label_length = np.full((batchsize, 1), 5)
        loss = ctc_batch_cost(y_true, y_pred, input_length, label_length)       
        return loss
    
    @tf.function
    def train_step(self, inputImages, targetLabels, batchsize):
        with tf.GradientTape(persistent=True) as tape:
            modelOutput = self.model(inputImages, training=True)
            loss = self.lossfunction(modelOutput, targetLabels, batchsize)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
    def train(self, train_X, train_y, epochs, batchsize, testImages=None, testLabels=None, checkpointInterval=10, verbose=True, pause_frequency=10, pause_duration=60):
        trainstart = time.time()
        n_batches = tf.data.experimental.cardinality(train_X).numpy()
        # iterate epochs
        for epoch in range(1, epochs + 1):
            epochstart = time.time()
            step = 0
            if verbose:
                progBar = tf.keras.utils.Progbar(n_batches)
            # iterate batches
            for inputBatch, targetBatch in tf.data.Dataset.zip((train_X, train_y)):
                self.train_step(inputBatch, targetBatch, batchsize)
                if verbose:
                    progBar.add(1)                
                step += 1
            ###
            # epoch finished
            # output epochtime
            if verbose:
                print("epoch %d took: %.2f seconds" % (epoch, time.time() - epochstart))
            # calculate current character accuracy rate
            if (not testImages is None) and (not testLabels is None):
                self.log_ar(testImages, testLabels, epoch)
            # save checkpoint at intervals
            if (not checkpointInterval is None) and (epoch % checkpointInterval) == 0:
                savepath = self.checkpoint_manager.save(checkpoint_number=epoch)
                print("saved to: {}".format(savepath))
                
            if (epoch % pause_frequency) == 0:
                time.sleep(pause_duration)
                
                
        # training finished
        if verbose:
            print("training took %.2f seconds" % (time.time() - trainstart) )
    ####
    # calculate character accuracy rate for current model.
    # save to csv-file.
    ####
    def log_ar(self, testImages, testLabels, epoch):
        if self.ar_file is None:
            return
        filepath = self.ar_file
        if not filepath.exists():
            filepath.touch()
            
        # calculate ar
        stime = time.time()
        ar = self.character_accuracyRate(testImages, testLabels,)
        print("ar(epoch=%d)=%f\t%fseconds" % (epoch, ar, time.time() - stime) )
        
        with filepath.open("a") as f:
            f.write("%d,%f\n" % (epoch, ar) )
          
    ####
    # Predicts labels from inputimages
    ####
    @tf.function
    def decode(self, inputImages, batchsize):
        modelOutput = self.model(inputImages, training=False)
        time_slices = self.input_shape[1]/8 
        input_length = np.full((batchsize), time_slices)
        decoded, probs = ctc_decode(modelOutput, input_length, greedy=True)
        labels =  tf.dtypes.cast(decoded[0], tf.int32)
        return labels
     
    """    
    ####
    # Transform label to sparsetensor of a shape suitable for tf.edit_distance()
    ####
    def _labelToSparse(self, labelTensor):
        values  = labelTensor.numpy()
        #values  = tf.make_ndarray(labelTensor)
        indices = [[0, 0, i] for i,_ in enumerate(values)]
        return tf.SparseTensor(indices, values, [1,1,1])    
    """
    
    ####
    # Transform dense tensor to sparse.
    ####
    def _l2s(self, dense, remove_blanks=True):
        if remove_blanks:
            indices = tf.where(tf.not_equal(dense, -1))
        else:
            indices = tf.where(tf.equal(dense, dense))
        result = tf.SparseTensor(indices, tf.gather_nd(dense, indices), tf.shape(dense, out_type=tf.int64))
        return result
    
    
    
    ####
    # calculate elementwise editdistances for two sets of labels
    ####
    @tf.function
    def labelDistance(self, pred, truth):
        # convert dense tensors to sparse.
        pred_s = self._l2s(pred)        
        truth_s = self._l2s(truth)        
        distance = tf.edit_distance(pred_s, truth_s, normalize=False)        
        return tf.cast(distance, tf.int32)
    
    ####
    # eliminate midstatedigits: either subtracts 10 from midstatedigit, or 9.5 if it is the last digit.
    ####
    def _decodeLabel(self, label):
        result = []
        for index, digit in enumerate(label.numpy()):
            if digit < 10:
                result.append(digit)
            elif index == len(label) - 1:
                result.append(digit - 9.5)
            else:
                result.append(digit - 10)
        return result
        
    
    @tf.function
    def character_accuracyRate(self, images_test, labels_test):           
        # predict labels from inputimages
        labels_result = self.decode(images_test, len(images_test))
        # count number of blanks (-1) in predicted labels
        #blanks = len(tf.where(tf.equal(labels_result, -1)))
        # calculate editdistance of each labelpair
        distances = self.labelDistance(labels_result, labels_test)
        # sum distances, subtract number of blanks
        sumDistances = tf.math.reduce_sum(distances)# - blanks
        # get number of charactes
        n_characters = tf.size(labels_test)
        # calculate character accuracy rate
        ar = 1 - (sumDistances / n_characters) 
        return ar
    
    """
    #@tf.function
    def character_accuracy_only(self, testDataset_x, testDataset_y, batchsize, n_characters, n_samples):
        sum_editdistance = 0
        # iterate batches
        for x_batch, y_batch in tf.data.Dataset.zip((testDataset_x, testDataset_y)):
            # predict labels
            #decoded, probs = self.decode(x_batch, batchsize)
            resultLabels = self.decode(x_batch, batchsize)#tf.dtypes.cast(decoded[0], tf.int32)
            
            # iterate every labelpair of this batch
            for result, truth in zip(resultLabels, y_batch):
                # get sparse tensors of labels
                result_sparse = self._labelToSparse(result)
                truth_sparse = self._labelToSparse(truth)
                # calc editdistance
                distance = tf.edit_distance(result_sparse, truth_sparse, normalize=False)
                # from this distance, subtract number of BLANKS (-1) used as padding
                distance -= tf.math.count_nonzero(result == -1, dtype=tf.dtypes.float32)
                # check whether whole line was correct
                #if distance == 0:
                #    total_correctLines += 1
                #else:
                    # check whether line was partially correct <=> whether line decodes to correct result
                #    if self._decodeLabel(result) == self._decodeLabel(truth):
                #        total_partiallyCorrectLines
                    # in case of any error, log labels and image.                   
                # add distance to total
                sum_editdistance += distance
        ar = 1 - (sum_editdistance/n_characters)
        #import pdb; pdb.set_trace()
        return ar
    """
    
    ####
    # calculates LCR (Line Correct Rate) and AR (Accuracy Rate)
    ####
    def accuracy(self, testDataset_x, testDataset_y, batchsize, n_characters, n_samples):
        # sum editdistance over every labelpair
        sum_editdistance = 0
        # count every correct line
        total_correctLines = 0
        total_partiallyCorrectLines = 0
        
        # keep track of labels with errors
        mismatches = []
        
        # iterate batches
        for x_batch, y_batch in tf.data.Dataset.zip((testDataset_x, testDataset_y)):
            # predict labels
            resultLabels = self.decode(x_batch, batchsize)            
            
            # iterate every labelpair of this batch
            for result, truth, image in zip(resultLabels, y_batch, x_batch):
                # get sparse tensors of labels
                #result_sparse = self._labelToSparse(result)
                #truth_sparse = self._labelToSparse(truth)
                # calc editdistance
                #distance = tf.edit_distance(result_sparse, truth_sparse, normalize=False)
                #distance = tf.cast(distance, tf.int32)
                distance = self.labelDistance(np.array([result]), np.array([truth]))[0]
                
                #import pdb; pdb.set_trace()
                # from this distance, subtract number of BLANKS (-1) used as padding
                #distance -= tf.math.count_nonzero(result == -1, dtype=tf.dtypes.float32)
                # check whether whole line was correct
                if distance == 0:
                    total_correctLines += 1
                else:
                    # check whether line was partially correct <=> whether line decodes to correct result
                    if self._decodeLabel(result) == self._decodeLabel(truth):
                        total_partiallyCorrectLines += 1
                    # in case of any error, log labels and image.
                    mismatches.append(
                        (result.numpy(), truth.numpy(), distance.numpy(), image.numpy())
                    )
                # add distance to total
                sum_editdistance += distance                        
        
        lcr = total_correctLines / n_samples        
        ar = 1 - (sum_editdistance/n_characters)
        lpr = (total_correctLines + total_partiallyCorrectLines) / n_samples
        return (lcr, ar, np.array(mismatches), lpr)            