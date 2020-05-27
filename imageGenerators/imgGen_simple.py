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
Args:
    digitImages
        contains for each digit (0-9) a list of images of that digit (possibly from different fonts) 
        digitImages.shape == (10, n_fonts, w,h,c)
"""
class synth_generator:
    def __init__(self, digitImages=None, vertical_margin = 20, digitsetPath="C:/Users/andre/Desktop/m/datasets/Chars74K/English/Fnt"):
        if digitImages==None:
            dsPath = Path(digitsetPath)
            self.digitImages = load_digits.load_char74k(dsPath, fonts=[28])
        else:
            self.digitImages = digitImages.copy()
        self.prepare_midstateDigits(verticalMargin = vertical_margin)

    
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

    
    
    ####
    # for each possible midstate (values 10-19), concatenate corresponding 2 digitimages vertically.
    # appends those to self.digitImages
    ####
    def prepare_midstateDigits(self, verticalMargin=10, padding_value=255):
        for midstate in range(10,20):
            # lower value digit, -> digit on top
            digit_lower = midstate - 10
            # higher value digit, -> digit on bottom
            digit_higher = digit_lower + 1 if midstate != 19 else 0
            
            image_lower = self.digitImages[digit_lower][0]
            image_higher = self.digitImages[digit_higher][0]
            
            # before concatenating images, remove margin that would result between them
            image_lower = self.cropImage(image_lower, top=False, right=False, left=False) # crop away bottom
            image_higher = self.cropImage(image_higher, bot=False, right=False, left=False) # crop away top
            # add custom vertical margin between images
            image_lower = cv2.copyMakeBorder(image_lower, top=0, bottom=verticalMargin, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=padding_value)
            
            midstate_image = cv2.vconcat([
                image_lower,
                image_higher
            ])
            self.digitImages.append([midstate_image])
            
    ####
    # generate a midstate character.
    ####
    # value: midstate-value between 10 and 20
    # y:     y-coordinate to take as center of window.
    #        between 0 and 1.
    #        y=0: result contains only lowervalue-digit
    #        y=1: result contains only highervalue-digit
    #        y>0 && y<1: result contains parts of both digits
    ####
    def midstateDigit(self, value, y):
        #assert value >= 10 and value < 20, "incorrect midstate-digitValue: %d; must be between 10 and 20" % (value)
        if value < 10:
            value += 10
        # when y is negative, "scroll backwards" a little
        if y < 0:
            value = value - 1 if value != 10 else 19
            y = 1 + y
        
        image = self.digitImages[value][0]
        initial_digitHeight = self.digitImages[0][0].shape[0] # size of window to crop from this image
        assert y >= 0 and y <= 1, "midstate-y must be between 0 and 1"
        y_center_topDigit = 0.5 * initial_digitHeight
        y_center_bottomDigit = image.shape[0] - 0.5 * initial_digitHeight
        y_center_window = y_center_topDigit + y * (y_center_bottomDigit - y_center_topDigit)
        # from center of window, define boundingbox
        bb_x = 0
        bb_width = image.shape[1]
        bb_y = int(y_center_window - 0.5*initial_digitHeight)
        bb_height = initial_digitHeight
        return image[bb_y : bb_y + bb_height, bb_x : bb_x + bb_width]
        
        
        
        
        
    
    
    
    ####
    # takes sequence of digits, loads their images and stitches them together.
    ####
    # inputs:
    #  digits-list<integer>: contains digits to use
    #  verticalShifts -list<float>: for each digit, how far it should be scrolled down (between -1,1)
    #  margins-list<integer>: for every digit (except the last), distance to his right neighbour. either single int or list of ints.
    #                length must be len(digits) - 1
    #  border-list<integer>: (top, bottom, left, right) padding of resultimage
    #  width, height - <integers>: target resolution to scale to (if both greater 0)
    #  font<integer>: index of the font to use for a given digit
    #  padding_value<integer>: intensity of added paddings and margins (usually white/255)
    #
    # range_normal,
    # range_midstate<2tuples>: range for how far digits can scroll vertically, between -1 and 1.
    #                          0 -> no scrolling, 0.5 -> scroll halfway to next("higher-value") digit, 1 -> scroll to next digit
    ####
    def generate_image(self, digits, margins, border, width=0, height=0, font=0, padding_value=255, draw_vertical_seperators=False, range_normal=(0,0), range_midstate=(0.5,0.5)):
        # if margins is single int, make list
        if isinstance(margins, int):
            margins = [margins for _ in range(0, len(digits) - 1)]
        
        assert (len(margins) == len(digits) - 1), "wrong number of margins. Expected: %d; Got: %d" % (len(digits) - 1, len(margins))
        fullmargins = margins.copy()  # make true copy of margins-list instead of using reference
        fullmargins.append(0) # add margin of 0 pixels to last digit to avoid edgeCaseHandling
        
        verticalShifts = [
            random.uniform(range_normal[0], range_normal[1]) if digit < 10 
            else random.uniform(range_midstate[0], range_midstate[1])
            for digit in digits        
        ]
        
        # for each digit, get image and apply vertical shift
        images = [
            self.midstateDigit(digit, vShift) for digit,vShift in zip(digits, verticalShifts)
        ]
        # crop images horizontaly
        images = [
            self.cropImageHorizontally(image) for image in images
        ]
        
        # to each image, apply padding to right neighbour
        images = [
            cv2.copyMakeBorder(digitImage, top=0, bottom=0, left=0, right=fullmargins[index], borderType=cv2.BORDER_CONSTANT, value=padding_value)
            for index, digitImage in enumerate(images) 
        ]
        
        # on each image, draw a vertical line inside the middle of padding to right neighbour        
        if draw_vertical_seperators:
            for image, margin in zip(images, fullmargins):
                if margin != 0:
                    x = int(image.shape[1] - margin/2)
                    cv2.line(image, (x,0), (x,image.shape[0]), color=0, thickness=1)
        
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
    
    ####
    # generates an image of digits for every sequence of digits in list_of_digits.
    # returns numpyarray.
    ####
    def generate_images(self, list_of_labels, margins, border, width=0, height=0, font=0, padding_value=255, draw_vertical_seperators=False, range_normal=(0,0), range_midstate=(0.5,0.5)):
        return np.array(
            [
               self.generate_image(label, margins, border, width, height, font, padding_value, draw_vertical_seperators, range_normal, range_midstate) 
               for label in list_of_labels
            ]
        )
        