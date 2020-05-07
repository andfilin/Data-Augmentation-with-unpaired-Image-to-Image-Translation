"""
functions for loading datasets to be used for modeltraining.
"""

from os import listdir
from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_COLOR, resize
from numpy.random import shuffle

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
def load_wmr(datasetPath, n_images=-1, imread_mode=IMREAD_GRAYSCALE,  shuffleImages=True, resize_to=None):
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
    