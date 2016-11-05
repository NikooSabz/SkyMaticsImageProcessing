import PIL
from sklearn import decomposition,cluster
import numpy as np
import glob
import cPickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
from shutil import copyfile
from skimage.feature import greycomatrix, greycoprops
from imfractal import *
from skimage import exposure
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel


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
    array = array.astype('int')
    values = k_means.cluster_centers_.squeeze()
    image_compressed = values[array].astype('uint8')
    image_compressed.shape = (size[0],size[1],3)
    test_image = PIL.Image.fromarray(image_compressed)
    return test_image
    
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
    
class_name = 'WhiteSpots'
base = '/home/geoanton/SkyMaticsLearning/FarmLabeledForTraining'
parent_dir = '%s/%s/' % (base,class_name)
new_folder = base + '/%sEnriched/' % class_name

if not(os.path.exists(new_folder)):
    os.makedirs(new_folder)
    
list_im = glob.glob(parent_dir + '*.png')
angles = [90,180,270,360]
i=0
for image_file in list_im:
    image = PIL.Image.open(image_file)
    name=class_name + '%04d' % i
    ext='png'
    for alpha in angles:
        print '%d out of %d' %(i,len(list_im))
        image.rotate(alpha).save(new_folder+'%s_%d.%s' %(name,alpha,ext))
    i+=1
    

list_im = glob.glob(new_folder + '*.png')
# Set up fractal functions:
ins = MFS()

ins.setDef(1,10,3)
# For GLCM
pixs = [2,4,8,16]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# for Gabor filters:
frqs = (0.1,0.5,1)
kernels = GetKernels(frqs)

reread_Hists=True
if reread_Hists:
    with open('kmeansadlee.pkl', 'rb') as fid:
        k_means = cPickle.load(fid)
    HistsMatrix = np.zeros((len(list_im),k_means.n_clusters))
    # Two features from the GLCM
    GLCMMatrix =  np.zeros((len(list_im),2*len(pixs)*len(angles)))
    FDMatrix= np.zeros((len(list_im),10))
    GaborFeats = np.zeros((len(list_im),len(frqs)*2*3))
    i=0
    for image_file in list_im:
        image = PIL.Image.open(image_file)
        arrayIm,greyImage=predictImageFromKmeans(k_means,image)
        ImagesMatrix = arrayIm
        HistsMatrix[i,:] = np.histogram(ImagesMatrix,bins=range(k_means.n_clusters+1))[0]
        GLCM  = greycomatrix(greyImage.astype('uint8'), pixs, angles, symmetric=True, normed=True)
        
        GLCMMatrix[i,:] = np.hstack((greycoprops(GLCM, 'dissimilarity').flatten(),greycoprops(GLCM, 'correlation').flatten()))
        FDMatrix[i,:]=ins.getFDs(image_file)
        GaborFeats[i,:] = GetFeaturesGaborFilters(greyImage,kernels).flatten()
        i+=1
        print 'Doing image number %d out of %d' %(i+1,len(list_im))
    np.savez(new_folder+ 'Feats%s.npz' % class_name,HistsMatrix = HistsMatrix.astype('int'),list_im = list_im,bins=range(k_means.n_clusters+1),
                                 GLCMMatrix=GLCMMatrix,FDMatrix = FDMatrix,GaborFeats = GaborFeats)
else:
    npzhists = np.load('HistsImagesFarm.npz')
    HistsMatrix = npzhists['HistsMatrix'].astype('int')
    GLCMMatrix = npzhists['GLCMMatrix']
    FDMatrix = npzhists['FDMatrix']
    GaborFeats =  npzhists['GaborFeats']
    list_im = npzhists['list_im']
    with open('kmeansadlee.pkl', 'rb') as fid:
        km_colors = cPickle.load(fid)

size = 65536.0
const = 0.2
# Reject black images (up to 25%)
inds = np.where(HistsMatrix[:,0]/size < const)[0]
HistsMatrix = HistsMatrix[inds,:].squeeze()
list_im = np.array(list_im)[inds]
        




