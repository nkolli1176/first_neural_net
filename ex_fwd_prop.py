#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:51:03 2019

Forward prop module for L layers
::: Linear  -> Relu -> Linear -> Relu -> .... -> Linear -> Sigmoid -> Output
@author: Naveen Kolli
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
    caches = {}
#    caches = []
    # Gotta save the inputs as the first activations
    caches["Z0"] = x0
#    print('Input size..'+str(x0.shape))

    for i in range(1, int(len(parameters.keys())/2)+1):
        zl = np.matmul(parameters['W'+str(i)], x0) + parameters['b'+str(i)]
        if (i == int(len(parameters.keys())/2)):
            # Output layer Activation is sigmoid        
            x0 = sigmoid(zl)
#            print('Fwd prop Layer..'+str(i)+'...Sigmoid..'+str(parameters['W'+str(i)].shape))            
        else:            
#            print('Fwd prop Layer..'+str(i)+'...Relu..'+str(parameters['W'+str(i)].shape))            
            x0 = relu(zl)
            
        caches["Z"+str(i)]=zl
        caches['W'+str(i)]=parameters['W'+str(i)]
        caches['b'+str(i)]=parameters['b'+str(i)]        

    AL = x0     
     
    return AL, caches


def main():
    # Uncomment to test the  function
    layers = [12288, 20, 13, 7, 5, 1] #  5-layer model
    
    params = ex_init_layer_weights.initialize_parameters_deep(layers)
    X = np.random.rand(layers[0],1000)
    #a = np.ones((3,2))
    #r = sigmoid(a)
    ##print(a)
    #print(r)
    ex_AL, ex_caches = L_model_forward(X, params)
    print(max(ex_AL.flatten()), min(ex_AL.flatten()), ex_AL.shape)
    print(ex_caches['Z0'].shape)

if __name__ == "__main__":
    main()

