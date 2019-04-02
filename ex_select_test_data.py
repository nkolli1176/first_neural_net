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
    img = np.array(img)
    print(img.shape)
    img = rgb2luma(img)
    plt.imshow(img)
    print(img.shape)
    
#    img_arr = np.asarray(img)/255
#    t_size = img_arr.shape
#
#    img_1d = img_arr.flatten()
#
#    reimg = Image.fromarray(np.reshape(img_1d,t_size)*255)
#    plt.imshow(reimg)
#

    return t_size

def rgb2luma(rgb):

    limg = 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]    
    return np.floor(limg)

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

def splitKfold(X, Y, j, kfolds, m):

    # K-fold xval, separate into train and x_val
    # If no kfold just use 10% as xval
    xval = int(m/kfolds)

    if (kfolds == 1):
        xval = int(m/10)
        x_train = X[:,0:m-xval]
        y_train = Y[0:m-xval]    
        x_xval = X[:,m-xval+1:]
        y_xval = Y[m-xval+1:]
    else:
        tmp_range = range(j*xval, (j+1)*xval)
        x_xval = X[:, tmp_range]
        y_xval = Y[tmp_range]
        # splits X and Y
        if (j==0):
            tmp_range = range((j+1)*xval+1, m)
            x_train = X[:, tmp_range]
            y_train = Y[tmp_range]
        else:
            tmp_range1 = range(0, j*xval)
            tmp_range2 = range((j+1)*xval+1, m)
            x_train = np.concatenate((X[:, tmp_range1], X[:, tmp_range2]), axis=1)
            y_train = np.concatenate((Y[tmp_range1], Y[tmp_range2]), axis=0)
            
    print(x_xval.shape, y_xval.shape, x_train.shape, y_train.shape)
    return x_xval, y_xval, x_train, y_train


def main():
    
    #trainfolder = '/Users/administrator/Downloads/Denoise/32/'
    #numtest = select_testImgs(trainfolder)
    #print(numtest)
#    X = np.loadtxt('dim_128'+'/X_K_test.dat')
#    Y = np.loadtxt('dim_128'+'/Y_K_test.dat')
#    m = X.shape[1]   
##    m = 1008
##    X = np.random.randn(512, m)
##    Y = np.random.randn(m,1)
#    kfolds = 5
#    
#    for i in range(0, kfolds):
#        X_xval, Y_xval, X_train, Y_train = splitKfold(X, Y, i, kfolds, m)
        
    imagename = '/Users/administrator/Documents/Python/Data_Kag/kaggle_train/cat.472.jpg'
    z = tryImage(imagename)
    print(z)
#    Y = np.zeros((1,10))+1
#    A = Y - 0.5
#    print(np.divide(1-Y,1-A))
    
if __name__ == "__main__":
    main()
