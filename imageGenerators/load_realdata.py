"""
functions for loading datasets to be used for modeltraining.
"""

from os import listdir
from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_COLOR, resize, BORDER_CONSTANT, copyMakeBorder
from numpy.random import shuffle
import pandas as pd
from pathlib import Path
import numpy as np

WMN_PATH = "C:/Users/andre/Desktop/m/datasets/SCUT-WMN DataSet"

####
# resize image.
# either return cv2.resize(image, dims) if ratio is not to be kept,
# else scale one dimension and pad other.
####
def resize_image(image, dims, keepRatio=False):
    if not keepRatio:
        return resize(image, dims)
    
    targetWidth = dims[0]
    targetHeight = dims[1]
    
    inputWidth = image.shape[1]
    inputHeight = image.shape[0]
    # scale either width or height, depending on which scaling factor would be smaller
    scale_width = targetWidth / inputWidth
    scale_height = targetHeight / inputHeight
    
    if scale_width < scale_height:
        # scale width, pad height
        result = resize(image, dsize=(0,0), fx=scale_width, fy=scale_width)
        padding = targetHeight - result.shape[0]
        p_top = int(padding/2)
        p_bot = p_top if (padding%2) == 0 else p_top + 1
        assert padding >= 0 and (p_top + p_bot) == padding, "unexpected height-padding: %d"%(padding)
        result = copyMakeBorder(result, top=p_top, bottom=p_bot, left=0, right=0, borderType=BORDER_CONSTANT,value=0)
    else:
        # scale height, pad width
        result = resize(image, dsize=(0,0), fx=scale_height, fy=scale_height)
        padding = targetWidth - result.shape[1]
        p_left = int(padding/2)
        p_right = p_left if (padding%2) == 0 else p_left + 1
        assert padding >= 0 and (p_left + p_right) == padding, "unexpected width-padding: %d"%(padding)
        result = copyMakeBorder(result, top=0, bottom=0, left=p_left, right=p_right, borderType=BORDER_CONSTANT,value=0)                                    
    return result

####
# For a given file of commaseperated imagepaths and labels, load images.
# processImage: function to apply to image before resizing (in-place)
####
def load_from_txt(txt_path, n_toLoad = None, seperators="[ ,]", resizeTo=None, keepRatio=False, imread_mode = IMREAD_GRAYSCALE, shuffleData=True, processImage=None):
    images = []
    labels = []
    df = pd.read_csv(txt_path, sep=seperators ,header=None)   
    if shuffleData:
        df = df.sample(frac=1).reset_index(drop=True)
        
    n_rows = df.shape[0]
    if n_toLoad is not None:
        n_rows = n_toLoad
        
    for i in range( n_rows ):
        row = df.values[i]
        imagepath = str(Path(WMN_PATH) / row[0])
        label = row[1:]
        image = imread(imagepath, imread_mode)
        if not processImage is None:
            processImage(image)
        if not resizeTo is None:
            image = resize_image(image, resizeTo, keepRatio)
        images.append(image)
        labels.append(label)
    return (np.array(images), np.array(labels).astype("int"))

def load_wmr_easy(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None):
    txt_path = Path(WMN_PATH) / "easy_samples.txt"
    return load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage)
def load_wmr_diff_train(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None):
    txt_path = Path(WMN_PATH) / "difficult_samples_for_train.txt"
    return load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage)
def load_wmr_diff_test(n_toLoad = None, resizeTo=None, keepRatio=False, processImage=None):
    txt_path = Path(WMN_PATH) / "difficult_samples_for_test.txt"
    return load_from_txt(txt_path, n_toLoad=n_toLoad, seperators="[ ,]", resizeTo=resizeTo, keepRatio=keepRatio, processImage=processImage)


# args:
#    datasetPath - <pathlib.Path>: path to dataset
#    n_images - <int>: number of images to load.
#                      if -1, load every image
#    imread_mode-<enum>: how images are to be opened. Default color(3-channels)
#    shuffle - <bool>: whether to shuffle images
#    resize_to <list[2]>: dimenisions to resize images to, if given. (width, height)
####
# returns:
#    list of len 2: loaded images, paths of images in same order
def load_wmr(datasetPath, n_images=-1, imread_mode=IMREAD_GRAYSCALE, shuffleImages=True, resize_to=None):
    imagePaths = [
        str(datasetPath / imageName) for imageName in listdir(datasetPath)
    ]
    if shuffleImages:
        shuffle(imagePaths)
    if n_images > 0:
        imagePaths = imagePaths[0:n_images]
    images = [
        imread(imagepath, imread_mode) for imagepath in imagePaths
    ]
    if resize_to != None:
        images = [
            resize(image, resize_to) for image in images
        ]
    
    return (images, imagePaths)
    