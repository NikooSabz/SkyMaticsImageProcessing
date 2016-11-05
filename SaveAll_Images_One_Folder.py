#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:02:46 2016

@author: geoanton
# Not a particularly important script, but helps saving all the images into one folder
It goes over all the folders within parent_dir and fetches the pngs, then saves them into a new folder.
new_folder : the destination
parent_dir : the sourceto go recursively over and fetch the pngs


"""


import fnmatch
import os
from shutil import copyfile

parent_dir = './images_half/'
matches = []
for root, dirnames, filenames in os.walk(parent_dir):
    for filename in fnmatch.filter(filenames, '*.png'):
        matches.append(os.path.join(root, filename))
        
new_folder = './images_all/'
if not(os.path.exists(new_folder)):
    os.makedirs(new_folder)
i=0
for item in matches:
    c,r = (item.split('/')[-1],item.split('/')[-2])
    fname = '%s_%s' % (r,c)
    dest = new_folder + fname
    copyfile(item,dest)
    print ' Done %d out of %d' %(i,len(matches))
    i+=1