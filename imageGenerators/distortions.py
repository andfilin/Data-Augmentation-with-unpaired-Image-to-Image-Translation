import cv2
import numpy as np


def brighten_blur(image, br_val, br_tresh, bl_size, bl_n, bl_kernel):
    brighten_digits(image, br_val, br_tresh)
    blur_squares(image, bl_size, bl_n, bl_kernel)
    
def blur_squares(image, size_rel, n, kernel_rel):
    k_absolute = int(image.shape[0] * kernel_rel)
    kernel = (k_absolute,k_absolute)
    for i in range(n):
        x0, y0, x1, y1 = _random_region(image, relative_heightRange=(size_rel,size_rel), relative_widthRange=(size_rel,size_rel))
        region = image[y0:y1, x0:x1]
        region = cv2.blur(region, kernel)
        image[y0:y1, x0:x1] = region[:, :]


def rotate(image, angle):
    
    fillColor = np.average(image)
    
    width = image.shape[1]
    height = image.shape[0]
    center = (width/2, height/2)
    Matrix = cv2.getRotationMatrix2D(center, angle, 1)
    result = cv2.warpAffine(image, Matrix, (width, height), borderValue=fillColor).astype("uint8")
    image[:,:] = result[:,:]

def saltPepperNoise(image, noiseAmount=0.1):
    width = image.shape[1]
    height = image.shape[0]
    
    noise = np.random.randint(0,2, (height, width)) * 255    
    mask = np.where( np.random.uniform(size=(height,width)) < noiseAmount, 1,0)
    result = np.where(mask, noise, image)
    image[:,:] = result[:,:]
    
def spNoise_region(image, noiseAmount=0.1):
    width = image.shape[1]
    height = image.shape[0]
    
    region_height = (0.2,0.8)
    region_width = (0.2,0.8)
    
    x0, y0, x1, y1 = _random_region(image, region_height, region_width)
    
    region = image[y0:y1, x0:x1]
    saltPepperNoise(region, noiseAmount)
    image[y0:y1, x0:x1] = region[:, :]

####
# return a random square-region (x0,y0,x1,y1) in an image, 
# with random width/height and position
####
def _random_region(image, relative_heightRange=(0,1), relative_widthRange=(0,1)):
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    
    minWidth = int(relative_widthRange[0] * imageWidth); maxWidth = int(relative_widthRange[1] * imageWidth)
    minHeight = int(relative_heightRange[0] * imageHeight); maxHeight = int(relative_heightRange[1] * imageHeight)
    
    if minWidth == maxWidth:
        regionWidth = minWidth
        regionHeight = minHeight
    else:
        regionWidth = np.random.randint(minWidth, maxWidth)
        regionHeight = np.random.randint(minHeight, maxHeight)

    x0 = np.random.randint(0, imageWidth - regionWidth)
    y0 = np.random.randint(0, imageHeight - regionHeight)
    x1 = x0 + regionWidth
    y1 = y0 + regionHeight
    
    return (x0, y0, x1, y1)
    
    
    

def blur_region(image, kernelRelativeToHeight=0.1):
    width = image.shape[1]
    height = image.shape[0]
    kernel = ( int(height*kernelRelativeToHeight)  )
    kernel = (kernel, kernel)
    
    # define random region to blur, relative to imageDims
    sizeRange = ( int(0.6 * height), int(0.9 * height) )
    minWidth = int(0.6 * width); maxWidth = int(0.9 * width)
    minHeight = int(0.6 * height); maxHeight = int(0.9 * height)
    blurWidth = np.random.randint(minWidth, maxWidth)
    blurHeight = np.random.randint(minHeight, maxHeight)
                 
    x0 = np.random.randint(0, width - blurWidth)
    y0 = np.random.randint(0, height - blurHeight)
    x1 = x0 + blurWidth
    y1 = y0 + blurHeight
    
    
    region = image[y0:y1,x0:x1]
    region = cv2.blur(region, kernel)
    image[y0:y1,x0:x1] = region
    
####
# draw <n_blots> circles in image.
# random size, positions, color
####
def add_round_blot(image, n_blots):
    width = image.shape[1]
    height = image.shape[0]
    average_value = np.average(image)
    n_channels = image.shape[-1]
    
    # minRadius = 3; maxRadius = 10
    # blot-radius relative to imageheight
    radiusRangeRelative = (0.1, 0.3)
    minRadius = int(height*radiusRangeRelative[0]); maxRadius = int(height*radiusRangeRelative[1])
    for _ in range(n_blots):    
        center = (
            np.random.randint(0,width),
            np.random.randint(0,height)
        )        
        
        radius = np.random.randint(minRadius, maxRadius)
        
        #color = average_value + np.random.randint(-20,20)            
        #color = 255 if np.random.random() < 0.5 else 0
        color = np.random.randint(0,255)
        if n_channels == 3:
            color = (color, color, color)
        
        cv2.circle(image, center, radius, color, thickness=-1, lineType=8 , shift=0)
        
####
# brighten digits (dark pixels).
# value: value to add to dark pixels
# treshold: biggest value to consider a digitpixel
####
def brighten_digits(image, value=40, treshold = 70):
    width = image.shape[1]
    height = image.shape[0]
    
    _, darkMap = cv2.threshold(cv2.bitwise_not(image), 255-treshold, 1, cv2.THRESH_BINARY_INV)
    darkMap = cv2.bitwise_not(darkMap) + 2
    #image[:,:] = darkMap[:,:]
    image[:,:] = cv2.addWeighted(image, 1, darkMap, value, 0)[:,:]
    """
    
    result = cv2.bitwise_not(image)
    
    result = result * 0.5
    result = result.astype("int8")
    
    result = cv2.bitwise_not(result)
    
    image[:,:] = result[:,:]
    
    
    " ""for colNumber in range(width):
        brightness_change = np.random.randint(-100,100)
        col = image[:,colNumber]
        col = np.array(col).astype("int16") + brightness_change
        col = np.clip(col, 0, 255).astype("int8")
        image[:,colNumber] = col
    """
    #patch = np.array(image[:, int(0.1*width):int(0.4*width)]).astype("int16") + 50 #* 1.5
    #import pdb; pdb.set_trace()
    #patch = np.clip(patch, 0, 255).astype("int8")
    #patch *= 0.5
    #image[:, int(0.1*width):int(0.4*width)] = patch
    