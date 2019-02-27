#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:06:55 2019

Back prop module, for logistic regression for now

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
#    print(m)
    da_next = np.sum(-np.divide(-Y[np.nonzero(Y==1)], AL[np.nonzero(Y==1)])) + np.sum(np.divide((1-Y[np.nonzero(Y==0)]), (1-AL[np.nonzero(Y==0)])))
    da_next = da_next/m
    glprime = 1
    
    for i in range(len(caches)-1, 0, -1):
        print(caches[i].shape)
#        dz = da_next + np.multiply(glprime, caches[i])
#        dW = 1/m * np.matmul(dz, relu(caches[i-1]))
#        db = 1/m * np.sum(dz)
#        da_prev = np.matmul(np.transpose(params["W"+str(i)]), dz)
#        
#        da_next = da_prev
#        grads["dW"+str(i)] = dW
#        grads["db"+str(i)] = db
            
        print(i)
        

        
    return grads
# Test the  function
layers = [12288, 20, 13, 7, 5, 1] #  4-layer model

params = ex_init_layer_weights.initialize_parameters_deep(layers)
X = np.random.rand(layers[0],)

ex_AL, ex_caches = ex_fwd_prop.L_model_forward(X, params)
print(ex_AL.shape)
print(len(ex_caches))
out = np.random.rand(1000,1)
y = np.floor(np.random.rand(1000,1)+0.5)
#c = ex_compute_cost.compute_cost(out, y)
#print(c)
grads = L_model_backward(out, y, ex_caches)
#print(len(grads))