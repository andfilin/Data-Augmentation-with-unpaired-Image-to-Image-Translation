import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from os import listdir
import numpy as np
import time
import pickle

"""
Generates synthetic images of metervalues by stitching together images of digits.
Args:
    digitImages
        contains for each digit (0-9) a list of images of that digit (possibly from different fonts) 
        digitImages.shape == (10, n_fonts, w,h,c)
"""
class synth_generator:
    def __init__(self, digitImages):
        self.digitImages = digitImages

    
    ####
    # Removes all padding from an image of a digit.
    # cv2.boundingRect assumes a white object in a black image
    # -> For black digit in white image, invert image when calculating boundingbox.
    ####
    def cropImage(self, image, invert=True):
        if invert:
            bb = cv2.boundingRect(cv2.bitwise_not(image))
        else:
            cv2.boundingRect(image)
        return image[bb[1] : bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
    ####
    # Removes white padding horizontaly (reduces width of image)
    ####
    def cropImageHorizontally(self, image, invert=True):
        if invert:
            bb = cv2.boundingRect(cv2.bitwise_not(image))
        else:
            cv2.boundingRect(image)
        # bounding box is a list: (x, y, width, height)
        # crop by slicing [y:y+h, x:x+w]
        return image[:, bb[0]:bb[0] + bb[2]]
    ####
    # Removes white padding vertically (reduces height of image)
    ####
    def cropImageVertically(self, image, invert=True):
        if invert:
            bb = cv2.boundingRect(cv2.bitwise_not(image))
        else:
            cv2.boundingRect(image)
        return image[bb[1] : bb[1] + bb[3], :]
    
    
    
    ####
    # takes sequence of digits, loads their images and stitches them together.
    ####
    # inputs:
    #  digits-list<integer>: contains digits to use
    #  margins-list<integer>: for every digit (except the last), distance to his right neighbour.
    #                length must be len(digits) - 1
    #  border-list<integer>: (top, bottom, left, right) padding of resultimage
    #  width, height - <integers>: target resolution to scale to (if both greater 0)
    #  font<integer>: index of the font to use for a given digit
    #  padding_value<integer>: intensity of added paddings and margins (usually white/255)
    ####
    def generate_image(self, digits, margins, border, width=0, height=0, font=0, padding_value=255):
        assert (len(margins) == len(digits) - 1), "wrong number of margins"
        margins = margins[:]  # make true copy of margins-list instead of using reference
        margins.append(0) # add padding of 0 pixels to last digit, avoids edgeCaseHandling
        
        # fetch images and crop them horizontaly
        images = [
            self.cropImageHorizontally(self.digitImages[digit][font]) for digit in digits
        ]
        
        # to each image, apply padding to right neighbour
        images = [
            cv2.copyMakeBorder(digitImage, top=0, bottom=0, left=0, right=margins[index], borderType=cv2.BORDER_CONSTANT, value=padding_value)
            for index, digitImage in enumerate(images) 
        ]
        
        # stitch all together
        result = cv2.hconcat(images)
        # remove vertical padding of result/reduce height
        result = self.cropImageVertically(result)
        
        # add specified padding to result
        result = cv2.copyMakeBorder(result, top=border[0], bottom=border[1], left=border[2], right=border[3], borderType=cv2.BORDER_CONSTANT, value=padding_value)
        
        # scale result
        if width > 0 and height > 0:
            result = cv2.resize(result, (width, height)) 
        
        return result