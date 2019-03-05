#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:22:50 2019
Read in train and test data from folder
Folder structure
test: folders 0 and 1 contain examples moved from the train set
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

def AssignImageData(folder):

    np.random.seed(1)
    im_1s = get_imlist(folder+'train/1')
    x_1s = []
    y_1s = []

    for im in im_1s:
#        myrand = np.random.rand()
        imh = np.array(Image.open(im))
        im1d = imh.flatten()
        x_1s.append(im1d)
        y_1s.append(np.ones(1,))

    im_0s = get_imlist(folder+'train/0')
    x_0s = []
    y_0s = []

    for im in im_0s:
        imh = np.array(Image.open(im))
        im1d = imh.flatten()
        x_0s.append(im1d)
        y_0s.append(np.zeros(1,))

    X_train = np.transpose(np.asarray(x_1s + x_0s)/255)
    Y_train = np.transpose(np.asarray(y_1s + y_0s))

    np.savetxt('X_train.dat', X_train)
    np.savetxt('Y_train.dat', Y_train)

    print(X_train.shape)
    print(Y_train.shape)
    print(Y_train[0,1])
    print(Y_train[0,12502])

    return X_train, Y_train

foldername = '/Users/nkolli/Documents/Python/Cats_vs_Dogs/'
X_tr, Y_tr = AssignImageData(foldername)
