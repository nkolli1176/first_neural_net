#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:06:55 2019

Back prop module, for logistic regression for now

AL: output activations for each of m vector inputs (1,m)
Y: labeled outputs for each of m vector inputs (1,m)
caches: 
    from Fwd prop, the Zs of each layer, including the input vector x0 (x0,m)
    Parameters W (nl+1), (nl)
    Parameters b (nl+1), 1
    So the count is numlayers*3 + 1, because the input Z stands alone withouta corresponding W and b
    
@author: Naveen Kolli
"""

import numpy as np
#import ex_compute_cost
import ex_fwd_prop
import ex_init_layer_weights

def relu(a):
    rel = np.maximum(a, 0)
    return rel


def L_model_backward(AL, Y, caches):

    grads = {}
    m = Y.shape[0]
    numlayers = int(np.floor(len(caches)/3))   
#    print(m)
    da_next = np.sum(-np.divide(-Y[np.nonzero(Y==1)], AL[np.nonzero(Y==1)])) + np.sum(np.divide((1-Y[np.nonzero(Y==0)]), (1-AL[np.nonzero(Y==0)])))
    da_next = da_next/m
#    print(numlayers)
    for i in range(numlayers, 0, -1):
        gprime = np.zeros(caches["Z"+str(i)].shape)
        gprime[np.nonzero(caches["Z"+str(i)])] = caches["Z"+str(i)][np.nonzero(caches["Z"+str(i)])]

        dz = np.multiply(da_next, gprime)
        dW = 1/m * np.matmul(dz, np.transpose(relu(caches["Z"+str(i-1)])))
        db = 1/m * np.sum(dz, axis=1, keepdims=True)
        da_prev = np.matmul(np.transpose(caches["W"+str(i)]), dz)
        print("Back prop Layer .."+str(i)+"...dW shape.."+str(dW.shape))
        
        da_next = da_prev
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

        
    return grads
# Test the  function
#layers = [12288, 20, 13, 7, 5, 1] #  5-layer model
#
#params = ex_init_layer_weights.initialize_parameters_deep(layers)
#X = np.random.rand(layers[0],10)
#
#ex_AL, ex_caches = ex_fwd_prop.L_model_forward(X, params)
##print(ex_AL.shape)
##print(len(ex_caches))
#out = np.random.rand(1000,1)
#y = np.floor(np.random.rand(1000,1)+0.5)
##c = ex_compute_cost.compute_cost(out, y)
##print(c)
#grads = L_model_backward(out, y, ex_caches)
##print(len(grads))
##for j in range(1, int(len(grads)/2)+1):
##    print(grads["dW"+str(j)].shape)
##    print(grads["db"+str(j)].shape)
#    
