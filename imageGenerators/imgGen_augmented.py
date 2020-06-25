# add current dir to syspath
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, current_dir)

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from os import listdir
import numpy as np
import time
import pickle
import random
#from imageGenerators\
import load_digits

"""
Generates synthetic images of metervalues by stitching together images of digits.
"""
class synth_generator:
    def __init__(self,  digitsetPath="C:/Users/andre/Desktop/m/datasets/Chars74K/English/Fnt", font=28):
        # load images of individual digits
        dsPath = Path(digitsetPath)
        digitImages = load_digits.load_char74k_singles(dsPath, font=font)  
        # crop digits
        digitImages = [self.cropImage(image) for image in digitImages]
        # normalize height and width of images by padding
        maxHeight = np.max([image.shape[0] for image in digitImages])
        maxWidth = np.max([image.shape[1] for image in digitImages])
        for i in range(len(digitImages)):
            image = digitImages[i]
            heightDiff = maxHeight - image.shape[0]
            widthDiff = maxWidth - image.shape[1]            
            
            topPadding = int(heightDiff/2)
            botPadding = topPadding if (heightDiff % 2) == 0 else topPadding + 1
            
            leftPadding = int(widthDiff/2)
            rightPadding = leftPadding if (widthDiff % 2) == 0 else leftPadding + 1
                    
            image = cv2.copyMakeBorder(image, 
                                           top=topPadding,
                                           bottom=botPadding,
                                           left=leftPadding,
                                           right=rightPadding,
                                           borderType=cv2.BORDER_CONSTANT, value=255)
            digitImages[i] = image
        assert np.max([image.shape[0] for image in digitImages]) == np.min([image.shape[0] for image in digitImages])
        assert np.max([image.shape[1] for image in digitImages]) == np.min([image.shape[1] for image in digitImages])
        
        
        self.digitHeight = maxHeight
        self.digitWidth = maxWidth
        
        # derive all sizes from digitHeight
        self.cellWidth = self.digitHeight
        self.cellHeight = int( 1.25*self.digitHeight )
        self.verticalMargin = int( 0.25 * self.digitHeight )
        self.horizontalMargin = int( (self.cellWidth -self.digitWidth) / 2 )
        self.borderWidth = int( 0.05 * self.digitHeight )
        
        # to each individual digitimage, append next higher digit at bottom
        imageZero = digitImages[0]
        for digit in range(len(digitImages)):
            image_top = digitImages[digit]
            image_bot = digitImages[digit + 1] if digit != 9 else imageZero
            # add margin at bottom and top of topImage
            image_top = cv2.copyMakeBorder(image_top,
                                           left=0, right=0,
                                           top=self.verticalMargin,
                                           bottom=self.verticalMargin,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=255)
            
            result = cv2.vconcat([
                image_top,
                image_bot
            ])
            digitImages[digit] = result
        
        self.digitImages = digitImages
        
       
    ####
    # label: array of digits to make image from
    # ranges: ranges of relative vertical offset for each digit (between 0 and 1)
    #            0 := No vertical offset, digit is centered
    #            1 := as much offset so that digit is not visible anymore 
    # offsets: vertical offsets for each digit. If None, generate randomly from ranges.
    ####
    def makeImage(self, label, normalstate_range=(-0.2,0.2), midstate_range=(0.3,0.7), resizeTo=None, color=True, rotate=True, offsets=None):

        # prepare one image for each digit
        cellImages = []        
        for position, digit in enumerate(label):
            
            # calculate random vertical offset            
            if digit < 10:
                y_range = normalstate_range                
            else:
                y_range = midstate_range
                digit -= 10     
                
            # if offsets given use them, else generate randomly from given ranges
            if offsets is None:
                y_relative = random.uniform(y_range[0], y_range[1])
            else:
                y_relative = offsets[position]
                
            if y_relative < 0:
                digit -= 1
                y_relative = 1 + y_relative
            
            # get imageslice according to vertical offset
            digitImage = self.digitImages[digit]
            bias = int(0.5*self.verticalMargin)
            y_offset = int(y_relative * (self.cellHeight-bias)) + bias
            digitImage = digitImage[y_offset:y_offset + self.cellHeight,:]            
            
            # define image of a single cell with border and insert image of digit
            cell = np.ones(shape=(self.cellHeight, self.cellWidth), dtype=int) * 255            
            cell_center_x = int(self.cellWidth/2)
            # insert digit into cell
            cell[:,self.horizontalMargin:self.horizontalMargin+self.digitWidth] = digitImage[:,:]
            # add border to cell
            cell = cv2.copyMakeBorder(cell, 
                               top=self.borderWidth,
                               bottom=self.borderWidth,
                               left=self.borderWidth,
                               right=0,
                               borderType=cv2.BORDER_CONSTANT, value=0)
            
            cellImages.append(cell)
        
        # stitch cells together
        result = cv2.hconcat(cellImages)
        # add final border to the rigth
        result = cv2.copyMakeBorder(result, 
                               top=0,
                               bottom=0,
                               left=0,
                               right=self.borderWidth,
                               borderType=cv2.BORDER_CONSTANT, value=0)
        
        if color:
            result = self.color_image(result)
        if rotate:
            result = self.rotateAndCrop(result)
        if not resizeTo is None:
            result = cv2.resize(result, resizeTo)
        
        return result
    
    def makeImages(self, label_list, normalstate_range=(-0.2,0.2), midstate_range=(0.3,0.7), resizeTo=None, color=True, rotate=True, includeChannels=True, offsets=None):
        #import pdb; pdb.set_trace()
        result = np.array([
            self.makeImage(label, normalstate_range, midstate_range, resizeTo, color, rotate, offsets[i]) for i, label in enumerate(label_list)
        ])
        if includeChannels:
            shape = [d for d in result.shape]
            shape.append(1)
            result = np.reshape(result, shape)
        return result    
        
    
    ####
    # taking a binary image as input (values 0 and 1),
    # color in different grayvalues.
    ####
    def color_image(self, image, range_black=(0,100), range_grey=(80,200), range_white=(180,255)):
        # get random shades of black,white,grey
        random_black = np.random.randint(range_black[0], range_black[1], image.shape)
        random_grey = np.random.randint(range_grey[0], range_grey[1], image.shape)
        # blur shades
        kernelsize = 7
        random_black = cv2.medianBlur(random_black.astype("uint8"), kernelsize)
        random_grey = cv2.medianBlur(random_grey.astype("uint8"), kernelsize)
        # apply shades
        result = image.copy()
        #result = np.where( (image != 0) and (image != 255), 255, result)
        result = np.where(image == 0, random_black, result)
        result = np.where(image == 255, random_grey, result)
        # blur result once again
        sigmaX = 1.5
        kernel = (3,3)
        result = cv2.GaussianBlur(result.astype("uint8"), kernel, sigmaX)
        return result
    
    def rotateAndCrop(self, image, maxAngle=3, crop=3):
        width = image.shape[1]; height = image.shape[0]
        center = (width/2, height/2)
        angle = np.random.randint(-maxAngle, maxAngle + 1)
        Matrix = cv2.getRotationMatrix2D(center, angle, 1)
        result = cv2.warpAffine(image.astype("uint8"), Matrix, (width, height), borderValue=255)
        return result[crop:-crop,crop:-crop]
    
    ####
    # Removes all padding from an image of a digit.
    # cv2.boundingRect assumes a white object in a black image
    # -> For black digit in white image, invert image when calculating boundingbox.
    ####
    def cropImage(self, image, invert=True, top=True, right=True, bot=True, left=True):
        if invert:
            bb = cv2.boundingRect(cv2.bitwise_not(image))
        else:
            bb = cv2.boundingRect(image)
        # bounding box is a list: (x, y, width, height)
        # crop by slicing [y:y+h, x:x+w]
        
        x, y, width, height = bb
   
        if not top:
            # dont crop top: y_new = 0; height += y_old
            height += y
            y = 0
        if not bot:
            # dont crop bot
            height = image.shape[0] - y   
        if not left:
            width += x
            x = 0
        if not right:
            width = image.shape[1] - x
        
        return image[y : y + height, x: x + width]
    
    ####
    # Removes white padding horizontaly (reduces width of image)
    ####
    def cropImageHorizontally(self, image, invert=True):
        return self.cropImage(image, invert, top=False, bot=False)

    ####
    # Removes white padding vertically (reduces height of image)
    ####
    def cropImageVertically(self, image, invert=True):
        return self.cropImage(image, invert, left=False, right=False)

    
    