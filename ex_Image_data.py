#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:22:50 2019
Read in train and test data from folder
Folder structure
test: folders 0 and 1 are empty
train: folders 0 and 1 contain classified examples
@author: administrator
"""

import numpy as np
from PIL import Image

import os

def get_imlist(path):
  """  Returns a list of filenames for
    all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def rgb2luma(rgb):

    limg = 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]    
    return np.floor(limg)

def AssignImageData(folder):

    np.random.seed(1)
    im_1s = get_imlist(folder+'test/1')
    x_1s = []
    y_1s = []
    
    for im in im_1s:    
#        myrand = np.random.rand()
        imh = np.array(Image.open(im))
        im1d = imh.flatten()
        x_1s.append(im1d)
        y_1s.append(np.ones(1,))

    im_0s = get_imlist(folder+'test/0')
    x_0s = []
    y_0s = []
    
    for im in im_0s:    
        imh = np.array(Image.open(im))
        im1d = imh.flatten()
        x_0s.append(im1d)
        y_0s.append(np.zeros(1,))

    X_train = np.transpose(np.asarray(x_1s + x_0s))
    Y_train = np.transpose(np.asarray(y_1s + y_0s))
    
    np.savetxt('X_test.dat', X_train)
    np.savetxt('Y_test.dat', Y_train)
    

    print(X_train.shape)
    print(Y_train.shape)
    print(Y_train[0,1])
    print(Y_train[0,2602])

#    print(len(X_train))
#    print(X_train[1].shape)
#    print(len(Y_train))
#    print(Y_train[1].shape)
#    print(Y_train[1])
#    print(Y_train[12502])
    
    train, test = [], []
    return train, test

def AssignLabelsImportImages(foldername, t_size):
    
    imgs = get_imlist(foldername)
    xs_train = []
    ys_train = []
    
    xs_test = []
    ys_test = []
    
    for im in imgs:
        myrand = np.random.rand()
        img = Image.open(im)
        img = img.resize((t_size), Image.ANTIALIAS)
        img = np.array(img)        
        # Convert to Luma, optional
        img = rgb2luma(img)
        
        if (myrand <= 0.2):
            xs_test.append(img.flatten())
            if ("cat" in os.path.basename(im)):
                ys_test.append(1)
            else:
                ys_test.append(0)
        else:
            xs_train.append(img.flatten())
            if ("cat" in os.path.basename(im)):
                ys_train.append(1)
            else:
                ys_train.append(0)
            
    X_train = np.transpose(np.asarray(xs_train))
    X_test = np.transpose(np.asarray(xs_test))
     
    Y_train = np.transpose(np.asarray(ys_train))
    Y_test = np.transpose(np.asarray(ys_test))

    return X_train, Y_train, X_test, Y_test

def main():
    foldername = '/Users/administrator/Documents/Python/Data_Kag/Kaggle_train/'
#    X, X_test = AssignImageData(foldername)
    t_size = (128, 128)
    xs_train, ys_train, xs_test, ys_test = AssignLabelsImportImages(foldername, t_size)
    print(xs_train.shape, ys_train.shape, xs_test.shape, ys_test.shape)

    np.savetxt('X_K_mono_test.dat', xs_test)
    np.savetxt('Y_K_mono_test.dat', ys_test)
    np.savetxt('X_K_mono_train.dat', xs_train)
    np.savetxt('Y_K_mono_train.dat', ys_train)

if __name__ == "__main__":
    main()
