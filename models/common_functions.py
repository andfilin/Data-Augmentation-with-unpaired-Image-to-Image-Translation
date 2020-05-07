import cv2
import numpy as np
from matplotlib import pyplot as plt
####
# creates an image comparing input and output of an generatormodel:
# args:
#    genModel:   model creating outputs out of testImages
#    testImages: numpy-array of images to feed to generator
####
def generate_comparisonImage(gen, testimages):
    # get numpyarray of samples
    samples = gen.predict(testimages)
    # transform testimages into numpyarray of images
    testimages = np.array([
        batch[0] for batch in testimages.as_numpy_iterator()
    ])
    
    # denormalize from -1,1 to 0,255
    samples = [image_preprocessing.denormalize(sample) for sample in samples]
    testimages = [image_preprocessing.denormalize(image) for image in testimages]
    
    # stitch all together
    inOut_pairs = []
    for index in range(len(samples)):
        input = testImages[0]
        output = samples[0]
        # add margin between pairs
        #input = cv2.copyMakeBorder(input, top=0, bottom=0, left=0, right=margin, borderType=cv2.BORDER_CONSTANT, value=255)
        # concatenate into pair
        inOut_pair = cv2.hconcat([input, output])
        # add margin below pair
        #inOut_pair = cv2.copyMakeBorder(inOut_pair, top=0, bottom=margin, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=255)
        inOut_pairs.append(inOut_pair)
    result = cv2.vconcat(inOut_pairs)
    # add margin to left, right and the top of result
    #result = cv2.copyMakeBorder(result, top=margin, bottom=0, left=margin, right=margin, borderType=cv2.BORDER_CONSTANT, value=255)
    return result


def plot_comparisonImage(gen, testimages, width, height, savepath):
    # get numpyarray of samples
    samples = gen.predict(testimages)
    samples = [
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in samples
    ]
    # transform testimages into numpyarray of images
    testimages = np.array([
        batch[0] for batch in testimages.as_numpy_iterator()
    ])
    
    # denormalize from -1,1 to 0,255
    #samples = [image_preprocessing.denormalize(sample) for sample in samples]
    #testimages = [image_preprocessing.denormalize(image) for image in testimages]
    n_images = len(testimages)
    fig, a = plt.subplots(n_images,2, figsize=(width,height), linewidth=1)
    for n in range(n_images):
        a[n][0].imshow(testimages[n], cmap='gray')
        a[n][0].axis("off")
        a[n][1].imshow(samples[n], cmap='gray')
        a[n][1].axis("off")
    
    a[0][0].set_title("Input")
    a[0][1].set_title("Output")
    #fig.show()
    fig.savefig(savepath)
    