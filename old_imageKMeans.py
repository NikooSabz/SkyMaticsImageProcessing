''' Out-dated script'''


import PIL
import glob
from sklearn import decomposition,cluster
import numpy as np
import cPickle
import matplotlib.pyplot as plt

n_clusters = 60
parent_dir = './images/'
images_list =  glob.glob(parent_dir + '*.png')
ind = range(len(images_list))
#ind=[84]
all_r = np.array([])
all_g = np.array([])
all_b = np.array([])

for i in ind:
    image = PIL.Image.open(images_list[i])
    imarray = np.array(image)
    all_r =np.hstack((all_r,imarray[:,:,0].flatten() ))
    all_g =np.hstack((all_g,imarray[:,:,1].flatten() ))
    all_b =np.hstack((all_b,imarray[:,:,2].flatten() ))
X = np.vstack((all_r,all_g,all_b)).T

k_means = cluster.KMeans(n_clusters=n_clusters, n_init=5)
k_means.fit(X)

values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
values = np.round(values).astype('int')
print ' KMeans done'
#image_compressed = [values[x,:] for x in labels]
image_compressed = values[labels].astype('uint8')
image_compressed = np.array(image_compressed)
image_compressed.shape = (all_r.shape[0]/256,256,3)
plt.figure(figsize = (5,10))
plt.imshow(image_compressed[:,:,:]/255.0)

n=6
with open('kmeansadlee.pkl', 'wb') as fid:
    cPickle.dump(k_means, fid)    

#with open('kmeansadlee.pkl', 'rb') as fid:
#    k_means = cPickle.load(fid)







# Show the prediction
test_image = PIL.Image.fromarray(image_compressed)
bg_w, bg_h = test_image.size
new_im = PIL.Image.new('RGB', (n*bg_w,bg_h))
w, h = new_im.size
# Iterate through a grid, to place the background tile
for i in xrange(0, w, bg_w):
    for j in xrange(0, h, bg_h):
      
        new_im.paste(test_image, (i, j))
new_im.save('Prediction%d.png' % n_clusters)

# Show the Original
X.shape = (all_r.shape[0]/256,256,3)
X=X.astype('uint8')

test_image = PIL.Image.fromarray(X)
bg_w, bg_h = test_image.size
new_im = PIL.Image.new('RGB', (n*bg_w,bg_h))
w, h = new_im.size
# Iterate through a grid, to place the background tile
for i in xrange(0, w, bg_w):
    for j in xrange(0, h, bg_h):
       
        new_im.paste(test_image, (i, j))
new_im.save('Original.png')
