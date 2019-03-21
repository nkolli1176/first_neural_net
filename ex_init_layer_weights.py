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
        parameters['W'+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i]) * np.sqrt(1/(layers_dims[i+1]*layers_dims[i]))
        parameters['b'+str(i+1)] = np.random.randn(layers_dims[i+1], 1) * np.sqrt(1/(layers_dims[i+1]))
#        print(parameters['W'+str(i+1)].shape)
#        print(parameters['b'+str(i+1)].shape)
        
    return parameters 

def main():
    layers = [1024, 20, 13, 5, 1]
    params = initialize_parameters_deep(layers)
    print(len(params))
    print(params.keys())
    print(params['W4'].shape)
    print(np.max(params['W3']))

#    num_layers = 5
#    parameters = {}
#    for i in range(num_layers-1):
#        parameters['W'+str(i+1)] = np.loadtxt('Data/Out_W'+str(i+1))
#        parameters['W'+str(i+1)] = np.asarray(parameters['W'+str(i+1)])
#        parameters['b'+str(i+1)] = np.loadtxt('Data/Out_b'+str(i+1))
#        parameters['b'+str(i+1)] = np.asarray(parameters['b'+str(i+1)])
#        
#        print(parameters['W'+str(i+1)].shape[0], parameters['W'+str(i+1)].shape[1])
#        print(parameters['b'+str(i+1)].shape)
    
if __name__ == "__main__":
    main()

