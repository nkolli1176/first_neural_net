#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:17:09 2019

# Script to move a subset of files to another directory
# Choose test set data from a large set of train Data

@author: administrator
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def tryImage(imname):
    img = Image.open(imname)
    t_size = (320,320)
    img = img.resize(t_size, Image.ANTIALIAS)
    plt.imshow(img)
    
#    img_arr = np.asarray(img)/255
#    t_size = img_arr.shape
#
#    img_1d = img_arr.flatten()
#
#    reimg = Image.fromarray(np.reshape(img_1d,t_size)*255)
#    plt.imshow(reimg)
#

    return t_size


def select_testImgs(folder):
    im_1s = get_imlist(folder+'train/0')
    numtest = 0
    for im in im_1s:
        myrand = np.random.rand()
        if (myrand < 0.2):
            os.rename(im, folder+'test/0/'+os.path.basename(im))
            numtest = numtest + 1
        else:
            continue


    return numtest

def main():
    
    #trainfolder = '/Users/administrator/Downloads/Denoise/32/'
    #numtest = select_testImgs(trainfolder)
    #print(numtest)
    
    imagename = '/Users/administrator/Downloads/Denoise/32/train/1/44.jpg'
    z = tryImage(imagename)
    print(z)
#    Y = np.zeros((1,10))+1
#    A = Y - 0.5
#    print(np.divide(1-Y,1-A))
    
if __name__ == "__main__":
    main()
