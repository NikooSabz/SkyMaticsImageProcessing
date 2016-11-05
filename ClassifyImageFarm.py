from sklearn import decomposition
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imfractal import *
from MiscFunctionsSkymatics import *

'''
In this script we try to train the classifier on the bank of labeled images.
Below, the images from three classes given in list_folder are read,
their feature vectors sitting in list_files.

X is then the feature matrix, Y - label matrix.


'''

# Read the DATA part
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
    
    


# Train the classifier part:

    
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




# Compute confusion matrix

# Plot non-normalized confusion matrix
class_names = list_files
# Plot normalized confusion matrix
ff =plt.figure()
plot_confusion_matrix(ConfMatrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


