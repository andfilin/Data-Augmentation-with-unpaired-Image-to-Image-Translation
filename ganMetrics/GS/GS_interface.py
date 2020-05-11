
from pathlib import Path
import pickle
import gs
from time import time
import numpy as np
from enum import Enum

# path where precalculated rtls for gs are to be saved
STATS_PATH = "C:/Users/andre/jupyter_ws/ganMetrics/GS/saved_statistics"
   
def imageset_to2D(imageset):
    new_shape = (imageset.shape[0], imageset.shape[1] * imageset.shape[2])
    return np.reshape(imageset, new_shape)
    
def store_rlts(rlts, savepath):
    print("saving to: %s" % (str(savepath)))
    with open(savepath, "wb") as picklefile:
        pickle.dump(rlts, picklefile)
        
def load_rlts(savepath):
    print("loading from: %s" % (str(savepath)))
    with open(savepath, "rb") as picklefile:
        rlts = pickle.load(picklefile)
    return rlts

def calculate_rlts(imageSet2D, n=1000, printTime=False):
    starttime = time()
    rlts = gs.rlts(imageSet2D, n=n)
    if printTime:
        print("calculating rlts took %f seconds" % (time() - starttime) )
    return rlts

####
# either loads rlts with given name if it exists,
# or calculates and saves it.
####
def load_or_calculateAndStore_rlts(imageSet2D, imagesetName, n=1000):
    saveName = make_savename(imageSet2D, imagesetName, n)
    savepath =  Path(STATS_PATH) / saveName
    if savepath.exists():
        rlts = load_rlts(savepath)
    else:
        rlts = calculate_rlts(imageSet2D, n=n)
        store_rlts(rlts, savepath)
    return rlts

####
# creates a filename containing dimensions of imageset and parameter n
####
def make_savename(imageSet2D, imagesetName, n):
    return "%s_%dx%d_n%d.pickle" % (imagesetName, imageSet2D.shape[0], imageSet2D.shape[1], n)

def score(rtls_A, rtls_B):
    return gs.geom_score(rtls_A, rtls_B)