"""
functions for loading digitimages to be used for imagegenerators.
"""

from os import listdir
import cv2
# loads from the char74k dataset.
# returns: array containing: for each digit array of images of that digit, possibly from different fonts. (for now: only one font) 
def load_char74k(datasetPath, imread_mode=cv2.IMREAD_GRAYSCALE, fonts=[28]):
    # get folder of every digit (Sample001 - Sample 010)
    digitFolders = [
        datasetPath / ("Sample00" + str(digit)) for digit in range(1,10)
    ]
    digitFolders.append(datasetPath / "Sample010")
    # map each digit to list of imagepaths
    digitImagePaths = []
    for digitFolder in digitFolders:
        imagePaths = [digitFolder / imageName for imageName in listdir(digitFolder)]
        digitImagePaths.append(imagePaths)    
    #n_fonts = 1 # number of fonts to load
    # open images 
    digitImages = [
        [cv2.imread(str(digitImagePaths[digit][fontindex]), imread_mode) for fontindex in fonts]
        for digit in range(10)
    ]
    return digitImages
   