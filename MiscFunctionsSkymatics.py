def predictImageFromKmeans(k_means,image,size=(256,256)):
    imarray = np.array(image)
    all_r =imarray[:,:,0].flatten()
    all_g =imarray[:,:,1].flatten() 
    all_b  = imarray[:,:,2].flatten() 
    X = np.vstack((all_r,all_g,all_b)).T
    newImage = k_means.predict(X)
    #values = k_means.cluster_centers_.squeeze()
    return newImage.astype('int')
    #values = np.round(values).astype('int')
    #image_compressed = values[newImage].astype('uint8')
    #image_compressed.shape = (size[0],size[1],3)

def GetImageFromKMeansAndArray(array,k_means,size=(256,256)):
    import PIL
    values = k_means.cluster_centers_.squeeze()
    image_compressed = values[array].astype('uint8')
    image_compressed.shape = (size[0],size[1],3)
    test_image = PIL.Image.fromarray(image_compressed)
    return test_image

