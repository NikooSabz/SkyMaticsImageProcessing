'''
Here we should put all of our elementary functions:
    predictImageFromKmeans - takes an image in rgb, returns its representation in k_means dictionary of colors,
    and its normalized greyscale image.
    GetImageFromKMeansAndArray - takes the kmeans representation and converts back to an RGB image, returns PIL image.


'''
from skimage import exposure
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
import PIL
import numpy as np

def predictImageFromKmeans(k_means,image,size=(256,256)):
    imarray = np.array(image)
    all_r =imarray[:,:,0].flatten()
    all_g =imarray[:,:,1].flatten() 
    all_b  = imarray[:,:,2].flatten() 
    X = np.vstack((all_r,all_g,all_b)).T
    newImage = k_means.predict(X)
    greyImage = 0.299*imarray[:,:,0] + 0.587 *imarray[:,:,1] + 0.114 * imarray[:,:,2]
    img_adapteq = exposure.equalize_hist(greyImage)
    image_grey_corr = (img_adapteq*255)


    return newImage.astype('int'),image_grey_corr


def GetImageFromKMeansAndArray(array,k_means,size=(256,256)):
    values = k_means.cluster_centers_.squeeze()
    image_compressed = values[array].astype('uint8')
    image_compressed.shape = (size[0],size[1],3)
    test_image = PIL.Image.fromarray(image_compressed)
    return test_image

'''
Gabor filter section:
    compute_feats - given the image and the Gabor kernels, computes the features
    GetFeaturesGaborFilters - wrapper for compute_feats. Use the wrapper instead of the compute_feats
    GetKernels - given the frequencies and angles, returns Gabor filters (their kernels)
    

'''
   
def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats    
        
def GetFeaturesGaborFilters(image,kernels):
    # image here is a gray-scaled image
    
    # prepare features
    ref_feats = np.zeros(( len(kernels), 2), dtype=np.double)
    ref_feats[:, :] = compute_feats(image, kernels)

    return ref_feats
    
def GetKernels(frqs =(0.05,0.1, 0.25,0.5),thetas = [0,0.5,0.75]):
    kernels = []
    thetas = np.array(thetas)*np.pi
    for theta in thetas:
      
        for frequency in frqs:
                kernel = np.real(gabor_kernel(frequency, theta=theta))
                kernels.append(kernel)
    return kernels
    
''' Machine learning misc functions:
    plot_confuision_matrix - given confusion matrix from predictions and class labels, plots the confusion matrix!

    '''

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%1.3f'% cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    