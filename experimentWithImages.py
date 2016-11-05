from __future__ import print_function

import PIL
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage import img_as_float
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage import feature




from scipy import ndimage as ndi

from skimage.filters import gabor_kernel


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

parent_dir = './images_farm/'
list_im = glob.glob(parent_dir + '*.png')
image_file = list_im[1200]
image = PIL.Image.open(image_file)

imarray = np.array(image)
greyImage = (0.299*imarray[:,:,0] + 0.587 *imarray[:,:,1] + 0.114 * imarray[:,:,2]).astype('uint8')


image_grey = PIL.Image.fromarray(greyImage)
img_adapteq = exposure.equalize_adapthist(greyImage, clip_limit=0.03)
image_grey_corr = PIL.Image.fromarray((img_adapteq*255).astype('uint8'))


thresh = threshold_otsu(img_adapteq)
binary = img_adapteq > thresh
binaryIm = PIL.Image.fromarray((binary*255).astype('uint8'))

canny = feature.canny(img_adapteq, sigma=5)
cannyIm = PIL.Image.fromarray((canny*255).astype('uint8'))


fig,ax = plt.subplots(nrows=2,ncols=4,figsize=(8,8))
plot_img_and_hist(image_grey,ax[:,0])
plot_img_and_hist(image_grey_corr,ax[:,1])
plot_img_and_hist(binaryIm,ax[:,2])
plot_img_and_hist(cannyIm,ax[:,3])

