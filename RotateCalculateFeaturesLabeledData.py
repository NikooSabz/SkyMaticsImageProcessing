''' QUite useless script, the only important part is the rotation, should be done in a separate script / function actually!

'''


import PIL

import numpy as np
import glob
import cPickle
import os
from shutil import copyfile
from skimage.feature import greycomatrix, greycoprops
from imfractal import *

from MiscFunctionsSkymatics import *
class_name = 'WhiteSpots'
base = '/home/geoanton/SkyMaticsLearning/FarmLabeledForTraining'
parent_dir = '%s/%s/' % (base,class_name)
new_folder = base + '/%sEnriched/' % class_name

if not(os.path.exists(new_folder)):
    os.makedirs(new_folder)
    
list_im = glob.glob(parent_dir + '*.png')
angles = [90,180,270,360]
i=0
# Here I save the copies of the initial images, but rotated by angle in angles:
for image_file in list_im:
    image = PIL.Image.open(image_file)
    name=class_name + '%04d' % i
    ext='png'
    for alpha in angles:
        print '%d out of %d' %(i,len(list_im))
        image.rotate(alpha).save(new_folder+'%s_%d.%s' %(name,alpha,ext))
    i+=1

    
    
    
# At this point now I go into the new_folder, and analyze images there treating each one as independent observation:

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
# This flag controls whether I have to re-build the feature sets or not.
# Rebuilding is an expensive procedure, although I get to re-run the script often sometimes. So make sure to set it to True when you 
# need to get the features, otherwise set it to False.
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
        




