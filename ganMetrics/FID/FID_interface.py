from pathlib import Path
import pickle
import fid
from time import time
import tensorflow.compat.v1 as tf_v1
#tf_v1.disable_v2_behavior()

"""
functions accessing official FID-Implementation to hide tensorflow-v1-stuff 
"""

# path where precalculated statistics for fib are to be saved
STATS_PATH = "C:/Users/andre/jupyter_ws/ganMetrics/FID/saved_statistics"

from enum import Enum
####
# enum of fib-statistics to be precalculated.
####
class SavedStatistics(Enum):
    WMN_EASY = "wmn_easy.pickle"
    WMN_DIFFICULT = "wmn_difficult.pickle"

# init inceptionmodel
inceptionPath = fid.check_or_download_inception(None)
fid.create_inception_graph(inceptionPath)    

default_batchsize = 100

###
# whether a fid-statistic defined in SavedStatistics-enum is already calculated
###
def stats_exist(statfile):
    path = Path(STATS_PATH) / statfile.value
    return path.exists()

###
# given an array of images, calculates fid-stats == (mu, sigma) used for calculating fid
###
def calculate_stats(imageset, batch_size=default_batchsize, printTime=False):
    starttime = time()
    with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(imageset, sess, batch_size=batch_size)
    if printTime:
        print("calculating rlts took %f seconds" % (time() - starttime) )
    return (mu, sigma)


####
# calculate and save fid-statistics for a given imageset from SavedStatistics-enum, if they dont exist already.
####
def init_precalculated_statistics(statfile, images, overwrite=False, batch_size=default_batchsize):
    assert isinstance(statfile, SavedStatistics), "argument to init_precalculated_statistics not an SavedStatistics-enum"
    path = Path(STATS_PATH) / statfile.value
    if not path.exists() or overwrite:
        print("calculating stats for %s and saving to %s" % (statfile.name, str(path)))
        mu, sigma = calculate_stats(images, batch_size)        
        with open(path, "wb") as picklefile:
            pickle.dump((mu, sigma), picklefile)
    else:
        print("stats for %s already exist" % (statfile.name))
    
####
# load fid-statistics precalculated with init_precalculated_statistics
####
def load_precalculated_statistics(statfile):
    assert isinstance(statfile, SavedStatistics), "argument to load_precalculated_statistics not an SavedStatistics-enum"
    path = Path(STATS_PATH) / statfile.value    
    assert path.exists(), "stats for %s are not initialized yet." % (statfile.name)
    # load stats
    with open(path, "rb") as picklefile:
        mu, sigma = pickle.load(picklefile)
    return (mu, sigma)

####
# calculate fid between 2 imagesets.
# inputs can be either image-array/s or SavedStatistics-enum/s
####
def calculate_fid(imagesA, imagesB, batch_size=default_batchsize):
    # load or calc stats for images_A
    if isinstance(imagesA, SavedStatistics):
        mu_A, sigma_A = load_precalculated_statistics(imagesA)
    else:
        mu_A, sigma_A = calculate_stats(imagesA, batch_size) 

    # load or calc stats for images_B            
    if isinstance(imagesB, SavedStatistics):
        mu_B, sigma_B = load_precalculated_statistics(imagesB)
    else:
        mu_B, sigma_B = calculate_stats(imagesB, batch_size) 
    # calculate fid
    fid_value = calculate_fid_from_stats( (mu_A, sigma_A), (mu_B, sigma_B) )
    return fid_value

def calculate_fid_from_stats(stats_A, stats_B):
    mu_A, sigma_A = stats_A
    mu_B, sigma_B = stats_B
    return fid.calculate_frechet_distance(mu_A, sigma_A, mu_B, sigma_B)
        
    