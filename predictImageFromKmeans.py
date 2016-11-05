''' Unsupervised clustering from the whole feature space 

This script helps labeling the images and sorting them into the folders for future supervised learning 

'''

import PIL
from sklearn import decomposition,cluster
import numpy as np
import glob
import cPickle
from sklearn.preprocessing import StandardScaler
import os
from shutil import copyfile
from skimage.feature import greycomatrix, greycoprops
from imfractal import *
from MiscFunctionsSkymatics import *
    
    
parent_dir = './images_all/'
list_im = glob.glob(parent_dir + '*.png')


# Set up fractal functions:
ins = MFS()

ins.setDef(1,10,3)

reread_Hists=False
if reread_Hists:
    with open('kmeansadlee.pkl', 'rb') as fid:
        k_means = cPickle.load(fid)
    HistsMatrix = np.zeros((len(list_im),k_means.n_clusters))
    # Two features from the GLCM
    GLCMMatrix =  np.zeros((len(list_im),2))
    FDMatrix= np.zeros((len(list_im),10))

    i=0
    for image_file in list_im:
        image = PIL.Image.open(image_file)
        arrayIm,greyImage=predictImageFromKmeans(k_means,image)
        ImagesMatrix = arrayIm
        HistsMatrix[i,:] = np.histogram(ImagesMatrix,bins=range(k_means.n_clusters+1))[0]
        GLCM  = greycomatrix(greyImage.astype('uint8'), [2], [0], 256, symmetric=True, normed=True)
        GLCMMatrix[i,:] = np.array([greycoprops(GLCM, 'dissimilarity')[0, 0],greycoprops(GLCM, 'correlation')[0, 0]])
        FDMatrix[i,:]=ins.getFDs(image_file)
        i+=1
        print 'Doing image number %d out of %d' %(i+1,len(list_im))
    
    
    np.savez('HistsImagesAll.npz',HistsMatrix = HistsMatrix.astype('int'),list_im = list_im,bins=range(k_means.n_clusters+1),
                                 GLCMMatrix=GLCMMatrix,FDMatrix = FDMatrix)
else:
    npzhists = np.load('HistsImagesAll.npz')
    HistsMatrix = npzhists['HistsMatrix'].astype('int')
    GLCMMatrix = npzhists['GLCMMatrix']
    FDMatrix = npzhists['FDMatrix']

    list_im = npzhists['list_im']
    with open('kmeansadlee.pkl', 'rb') as fid:
        km_colors = cPickle.load(fid)
size = 65536.0
const = 0.2
# Reject black images (up to 25%)
inds = np.where(HistsMatrix[:,0]/size < const)[0]
HistsMatrix = HistsMatrix[inds,:].squeeze()
list_im = list_im[inds]
        

X = np.hstack((HistsMatrix,GLCMMatrix[inds,:].squeeze(),FDMatrix[inds,:].squeeze()))
        
#HistsMatrix = StandardScaler().fit_transform(HistsMatrix)
X= StandardScaler().fit_transform(X)
n_clusters = 15
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
    r,c = list_im[i].split('/')[-1].split('_')
    new_name = new_folder+"C%d_%s_%s" %(labels[i],r,c)
    copyfile(fname,new_name)
    print ' Done %d out of %d' %(i,len(list_im))

pca = decomposition.PCA(n_components=2)
XX= pca.fit_transform(X)






