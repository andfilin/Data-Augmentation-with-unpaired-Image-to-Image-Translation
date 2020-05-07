import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# resize image and add padding on one dimensions to keep ratios
def resize_withPadding(image, targetWidth, targetHeight):    
    inputWidth = image.shape[1]
    inputHeight = image.shape[0]
    # scale either width or height, depending on which scaling factor would be smaller
    scale_width = targetWidth / inputWidth
    scale_height = targetHeight / inputHeight
    
    if scale_width < scale_height:
        # scale width, pad height
        result = cv2.resize(image, dsize=(0,0), fx=scale_width, fy=scale_width)
        padding = targetHeight - result.shape[0]
        p_top = int(padding/2)
        p_bot = p_top if (padding%2) == 0 else p_top + 1
        assert padding >= 0 and (p_top + p_bot) == padding, "unexpected height-padding: %d"%(padding)
        result = cv2.copyMakeBorder(result, top=p_top, bottom=p_bot, left=0, right=0, borderType=cv2.BORDER_CONSTANT,value=0)
    else:
        # scale height, pad width
        result = cv2.resize(image, dsize=(0,0), fx=scale_height, fy=scale_height)
        padding = targetWidth - result.shape[1]
        p_left = int(padding/2)
        p_right = p_left if (padding%2) == 0 else p_left + 1
        assert padding >= 0 and (p_left + p_right) == padding, "unexpected width-padding: %d"%(padding)
        result = cv2.copyMakeBorder(result, top=0, bottom=0, left=p_left, right=p_right, borderType=cv2.BORDER_CONSTANT,value=0)                                    
    return result

# open and resize an image
def loadImage(imgPath, width, height, keepRatio=True):
    image = cv2.imread(str(imgPath), cv2.IMREAD_COLOR)
    if keepRatio:
        image = resize_withPadding(image, width, height)
    else:
        image = cv2.resize(image, (width, height))
    assert image.shape[0] == height and image.shape[1] == width, "resizing failed"
    return image


# read wmn-train/testdatafiles containing imagepaths and labels, return images and labels 
def load_wmn_traindata(width, height, datasetPath = "C:/Users/andre/Desktop/m/datasets/SCUT-WMN DataSet", keepRatio = True):
    # load data
    if not isinstance(datasetPath, Path):
        datasetPath = Path(datasetPath)
    trainfile = datasetPath / "difficult_samples_for_train.txt"
    testfile = datasetPath / "difficult_samples_for_test.txt"
    
    # load trainimages
    df_train = pd.read_csv(trainfile, sep="[ ,]" ,header=None)
    
    images_train = np.array([
        loadImage(datasetPath / row[0], width, height, keepRatio) for row in df_train.values
    ]).astype("float32")
    labels_train = np.array([
        row[1:] for row in df_train.values
    ]).astype("int")
    
    # load testimages
    df_test = pd.read_csv(testfile, sep="[ ,]" ,header=None)
    
    images_test = np.array([
        loadImage(datasetPath / row[0], width, height, keepRatio) for row in df_test.values
    ]).astype("float32")
    labels_test = np.array([
        row[1:] for row in df_test.values
    ]).astype("int")
    
    return (images_train, labels_train, images_test, labels_test)

def load_easySamples(width, height, datasetPath = "C:/Users/andre/Desktop/m/datasets/SCUT-WMN DataSet", keepRatio=True):
    # load data
    if not isinstance(datasetPath, Path):
        datasetPath = Path(datasetPath)
    file = datasetPath / "easy_samples.txt"
    df = pd.read_csv(file, sep="[ ,]" ,header=None)
    
    images = np.array([
        loadImage(datasetPath / row[0], width, height, keepRatio) for row in df.values
    ]).astype("float32")
    labels = np.array([
        row[1:] for row in df.values
    ]).astype("int")
    
    return (images, labels)