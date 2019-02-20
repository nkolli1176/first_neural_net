#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 21:06:24 2019

Takes layer dimensions for a NN and initializes parameters of the right sizes
NEEDS UPDATING: Fully connected layers, last one is the output layer with size 1 (needs updating)
Normal random distribution is used for sampling the parameters

@author: Naveen Kolli
"""
import numpy as np
#import matplotlib as plt
def initialize_parameters_deep(layers_dims):

    num_layers = len(layers_dims)
#    print(num_layers)
    parameters = {}
    
    for i in range(num_layers-1):
        parameters['W'+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i])
        parameters['b'+str(i+1)] = np.random.randn(layers_dims[i+1], )
#        print(parameters['W'+str(i+1)].shape)
#        print(parameters['b'+str(i+1)].shape)
        
    return parameters 

