#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:51:03 2019

@author: administrator
"""
import numpy as np
import ex_init_layer_weights

def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig

def relu(a):
    rel = np.maximum(a, 0)
    return rel

def L_model_forward(X, parameters):
    x0 = X
    caches = []
    
    for i in range(1, int(len(params.keys())/2)+1):
        print('Processing Layer..'+str(i))
        al = np.matmul(parameters['W'+str(i)], x0) + parameters['b'+str(i)]
        x0 = relu(al)
        
        caches.append(x0)
        
    AL = sigmoid(x0)    
    caches.append(AL)
    # Output layer here is sigmoid
    
    return AL, caches

# Test the  function
layers = [12288, 20, 13, 7, 5, 1] #  4-layer model

params = ex_init_layer_weights.initialize_parameters_deep(layers)
X = np.random.rand(layers[0],)
#a = np.ones((3,2))
#r = sigmoid(a)
##print(a)
#print(r)
ex_AL, ex_caches = L_model_forward(X, params)
print(ex_AL.shape)
print(len(ex_caches))