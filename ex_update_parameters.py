#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:46:43 2019

@author: administrator
"""
import numpy as np
import ex_init_layer_weights
import ex_fwd_prop
import ex_back_prop

def update_parameters(parameters, grads, learning_rate):

    for i in range(1, int(len(parameters.keys())/2)+1):
        print('Updating Layer..'+str(i))
        parameters['W'+str(i)] = parameters['W'+str(i)] - (learning_rate * grads['dW'+str(i)])
        parameters['b'+str(i)] = parameters['b'+str(i)] - (learning_rate * grads['db'+str(i)])

    return parameters

# Test the  function
layers = [12288, 20, 13, 7, 5, 1] #  4-layer model

params = ex_init_layer_weights.initialize_parameters_deep(layers)
X = np.random.rand(layers[0],10)

ex_AL, ex_caches = ex_fwd_prop.L_model_forward(X, params)
out = np.random.rand(1000,1)
y = np.floor(np.random.rand(1000,1)+0.5)
grads = ex_back_prop.L_model_backward(out, y, ex_caches)

newparams = update_parameters(params, grads, 0.008)
