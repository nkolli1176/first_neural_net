#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:51:03 2019

Forward prop module for L layers
::: Linear  -> Relu -> Linear -> Relu -> .... -> Linear -> Sigmoid -> Output
@author: Naveen Kolli
"""
import numpy as np
#import ex_init_layer_weights

def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig

def relu(a):
    rel = np.maximum(a, 0)
    return rel

def L_model_forward(X, parameters):
#    x0 = X
#    caches = []
##    caches = []
#    # Gotta save the inputs as the first activations
#    caches.append(x0)
#
#    for i in range(1, int(len(parameters.keys())/2)+1):
#        print('Fwd prop Layer..'+str(i)+'...shape'+str(parameters['W'+str(i)].shape))
#        zl = np.matmul(parameters['W'+str(i)], x0) + parameters['b'+str(i)]
#        x0 = relu(zl)
#        caches.append(zl)
#
#    # Output layer Activation is sigmoid        
#    AL = sigmoid(x0)    

    x0 = X
    caches = {}
#    caches = []
    # Gotta save the inputs as the first activations
    caches["Z0"]=x0

    for i in range(1, int(len(parameters.keys())/2)+1):
        print('Fwd prop Layer..'+str(i)+'...shape'+str(parameters['W'+str(i)].shape))
        zl = np.matmul(parameters['W'+str(i)], x0) + parameters['b'+str(i)]
        x0 = relu(zl)
        caches["Z"+str(i)]=x0
        caches['W'+str(i)]=parameters['W'+str(i)]
        caches['b'+str(i)]=parameters['b'+str(i)]        

    # Output layer Activation is sigmoid        
    AL = sigmoid(x0)    
     
    return AL, caches

## Uncomment to test the  function
#layers = [12288, 20, 13, 7, 5, 1] #  5-layer model
#
#params = ex_init_layer_weights.initialize_parameters_deep(layers)
#X = np.random.rand(layers[0],)
##a = np.ones((3,2))
##r = sigmoid(a)
###print(a)
##print(r)
#ex_AL, ex_caches = L_model_forward(X, params)
#print(ex_AL.shape)
#print(len(ex_caches))