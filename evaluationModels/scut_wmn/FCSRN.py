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
    
    def __init__(self, input_shape, checkpoint_path = None):
        self.input_shape = input_shape
        # build model
        input, convNet_output = convNet(input_shape)
        output = temporalMapper(convNet_output)
        output = Activation("softmax")(output)
        self.model = tf.keras.Model(inputs=input, outputs=output)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=0.0001)
        # prepare checkpoint
        if checkpoint_path != None:
            self.checkpoint = tf.train.Checkpoint(
                model = self.model,
                optimizer = self.optimizer,
            )
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=None, checkpoint_name="epoch")
            # load existing checkpoint if it exists
            if self.checkpoint_manager.latest_checkpoint:
                checkpoint_to_be_loaded = self.checkpoint_manager.latest_checkpoint
                self.checkpoint.restore(checkpoint_to_be_loaded)
                print("loaded checkpoint: ", checkpoint_to_be_loaded)
            else:
                print("created new Model")
        else:
            self.checkpoint = None
            print("No checkpointpath given, model will not be saved.")
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
        
    def train(self, train_X, train_y, epochs, batchsize):
        for epoch in range(epochs + 1):
            epochstart = time.time()
            step = 0
            for inputBatch, targetBatch in tf.data.Dataset.zip((train_X, train_y)):
                stepstart = time.time() 
                self.train_step(inputBatch, targetBatch, batchsize)
                print("step %d took: %f seconds" % (step, time.time() - stepstart))
                step += 1
            print("epoch %d took: %f seconds" % (epoch, time.time() - epochstart))
        savepath = self.checkpoint_manager.save(checkpoint_number=epochs)
        print("saved to: {}".format(savepath))
          
    ####
    # Predicts labels from inputimages
    ####
    def decode(self, inputImages, batchsize):
        modelOutput = self.model(inputImages, training=False)
        time_slices = self.input_shape[1]/8 
        input_length = np.full((batchsize), time_slices)
        return ctc_decode(modelOutput, input_length, greedy=True)
    
    ####
    # Transform label to sparsetensor of a shape suitable for tf.edit_distance()
    ####
    def _labelToSparse(self, labelTensor):
        values  = labelTensor.numpy()
        indices = [[0, 0, i] for i,_ in enumerate(values)]
        return tf.SparseTensor(indices, values, [1,1,1])
    
    ####
    # calculates LCR (Line Correct Rate) and AR (Accuracy Rate)
    ####
    def accuracy(self, testDataset_x, testDataset_y, batchsize, n_characters, n_samples):
        # sum editdistance over every labelpair
        sum_editdistance = 0
        # count every correct line
        total_correctLines = 0
        
        # keep track of labels with errors
        mismatches = []
        
        # iterate batches
        for x_batch, y_batch in tf.data.Dataset.zip((testDataset_x, testDataset_y)):
            # predict labels
            decoded, probs = self.decode(x_batch, batchsize)
            resultLabels = tf.dtypes.cast(decoded[0], tf.int32)
            
            # iterate every labelpair of this batch
            for result, truth, image in zip(resultLabels, y_batch, x_batch):
                # get sparse tensors of labels
                result_sparse = self._labelToSparse(result)
                truth_sparse = self._labelToSparse(truth)
                # calc editdistance
                distance = tf.edit_distance(result_sparse, truth_sparse, normalize=False)
                # from this distance, subtract number of BLANKS (-1) used as padding
                distance -= tf.math.count_nonzero(result == -1, dtype=tf.dtypes.float32)
                # check whether whole line was correct
                if distance == 0:
                    total_correctLines += 1
                else:
                    mismatches.append(
                        (result.numpy(), truth.numpy(), distance.numpy(), image.numpy())
                    )
                # add distance to total
                sum_editdistance += distance
                
        lcr = total_correctLines / n_samples        
        ar = 1 - (sum_editdistance/n_characters)
        return (lcr, ar, np.array(mismatches))            