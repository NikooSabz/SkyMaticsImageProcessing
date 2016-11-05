''' Unsupervised clustering of the farm images

Can be done after the unsupervised clustering of the whole image and deleting the outliers. Helps find some repetitive patterns 
in the images describing the farm land given that those sit in images_farm folder.

'''

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
from MiscFunctionsSkymatics import *

    
parent_dir = './images_farm/'
list_im = glob.glob(parent_dir + '*.png')
list_im = list_im[::3]


# Set up fractal functions:
ins = MFS()

ins.setDef(1,10,3)
# For GLCM
pixs = [2,4,8,16]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# for Gabor filters:
frqs = (0.1,0.5,1)
kernels = GetKernels(frqs)



reread_Hists=False
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
    
    
    np.savez('HistsImagesFarm.npz',HistsMatrix = HistsMatrix.astype('int'),list_im = list_im,bins=range(k_means.n_clusters+1),
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
        

#X = np.hstack((HistsMatrix,GLCMMatrix[inds,:].squeeze(),FDMatrix[inds,:].squeeze()))
X = np.hstack((HistsMatrix,GLCMMatrix[inds,:].squeeze(),GaborFeats[inds,:].squeeze(),FDMatrix[inds,:].squeeze()))
#HistsMatrix = StandardScaler().fit_transform(HistsMatrix)
X= StandardScaler().fit_transform(X)
n_clusters = 2
k_means_hist = cluster.KMeans(n_clusters=n_clusters, n_init=20)
#k_means_hist.fit(HistsMatrix)
k_means_hist.fit(X)
labels = k_means_hist.labels_
print ' KMeans done'

new_folder = './images_all_Marked/'
if not(os.path.exists(new_folder)):
    os.makedirs(new_folder)
for i in range(0,len(list_im),1):
    fname = list_im[i]
    new_name = new_folder+"C%d_%d.png" %(labels[i],i)
    copyfile(fname,new_name)
    print ' Done %d out of %d' %(i,len(list_im))

fig,ax = plt.subplots()
pca = decomposition.PCA(n_components=2)
XX= pca.fit_transform(X)

ax.scatter(XX[:,0],XX[:,1],c=labels,cmap='jet')
ax.set_xlim([-15,10])
ax.set_ylim([-10,10])





