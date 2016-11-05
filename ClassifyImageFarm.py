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

base = '/home/geoanton/SkyMaticsLearning/FarmLabeledForTraining/'
list_folder = [base +x for x in ['PlowedEnriched/','WhiteSpotsEnriched/','RoadsEnriched/']]
list_files = ['FeatsPlowed.npz','FeatsWhiteSpots.npz','FeatsRoads.npz']
n_feats=110
X_all=np.empty(n_feats)
Y_all=np.empty(1)
for i in range(3):
    npzfeats = np.load(list_folder[i]+list_files[i])
    HistsMatrix = npzfeats['HistsMatrix'].astype('int')
    GLCMMatrix = npzfeats['GLCMMatrix']
    FDMatrix = npzfeats['FDMatrix']
    GaborFeats =  npzfeats['GaborFeats']
    X = np.hstack((HistsMatrix,GLCMMatrix[:,:].squeeze(),GaborFeats[:,:].squeeze(),
                   FDMatrix[:,:].squeeze()))
    Y = np.ones(X.shape[0])*i
    X_all = np.vstack((X_all,X))
    Y_all= np.hstack((Y_all,Y.T))
Y_all[0]=0
    
        
#X = np.hstack((HistsMatrix,GLCMMatrix[inds,:].squeeze(),FDMatrix[inds,:].squeeze()))
#HistsMatrix = StandardScaler().fit_transform(HistsMatrix)
X= StandardScaler().fit_transform(X_all)
Y=Y_all

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
logreg = LogisticRegressionCV()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

ConfMatrix  = metrics.confusion_matrix(y_test,y_pred)
ScoreMetric =  metrics.accuracy_score(y_test, y_pred)
print ScoreMetric,"\n",ConfMatrix

fig,ax = plt.subplots()
pca = decomposition.PCA(n_components=2)
XX= pca.fit_transform(X)

ax.scatter(XX[:,0],XX[:,1],c=Y_all,cmap='jet',vmin=0,vmax=2)
ax.set_xlim([-15,10])
ax.set_ylim([-10,10])



import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

# Compute confusion matrix

# Plot non-normalized confusion matrix
class_names = list_files
# Plot normalized confusion matrix
ff =plt.figure()
plot_confusion_matrix(ConfMatrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


