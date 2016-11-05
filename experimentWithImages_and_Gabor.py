from __future__ import print_function

'''Script for experimenting purposes. Good place to test some ideas etc.'''


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
image_file = list_im[2000]
image = PIL.Image.open(image_file)

imarray = np.array(image)
greyImage = (0.299*imarray[:,:,0] + 0.587 *imarray[:,:,1] + 0.114 * imarray[:,:,2]).astype('uint8')


image_grey = PIL.Image.fromarray(greyImage)
img_adapteq = exposure.equalize_adapthist(greyImage, clip_limit=0.03)
image_grey_corr = PIL.Image.fromarray((img_adapteq*255).astype('uint8'))


thresh = threshold_otsu(img_adapteq)
binary = img_adapteq > thresh
binaryIm = PIL.Image.fromarray((binary*255).astype('uint8'))

fig,ax = plt.subplots(nrows=2,ncols=4,figsize=(8,8))
plot_img_and_hist(image_grey,ax[:,0])
plot_img_and_hist(image_grey_corr,ax[:,1])
plot_img_and_hist(binaryIm,ax[:,2])




def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats



# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for frequency in (0.05,0.1, 0.25,0.5):
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            kernels.append(kernel)


brick = img_adapteq
image_names = ('sample')
images = (brick,)

# prepare reference features
ref_feats = np.zeros((1, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(brick, kernels)




def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for frequency in (0.05,0.1, 0.25,0.5):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nf=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=len(results), ncols=2, figsize=(2, 20))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
fig.savefig('gabor.png',dpi=300)