#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:06:55 2019

Back prop module, for logistic regression for now

AL: output activations for each of m vector inputs (1,m)
Y: labeled outputs for each of m vector inputs (1,m)
caches: 
    from Fwd prop, 
    As of each layer, including the input vector x0 (x0,m)
    Zs of each layer
    Parameters W (nl+1), (nl)
    Parameters b (nl+1), 1
    So the count is numlayers*4 + 1, because the input A0 stands alone 
    
@author: Naveen Kolli
"""

import numpy as np
import ex_compute_cost
import ex_fwd_prop
import ex_init_layer_weights
import ex_activations

def L_model_backward(AL, Y, caches):

    grads = {}
    m = len(Y.flatten())
    # Condition the shapes so that output and labels align
    Y = np.reshape(Y, (1,m))
    AL = np.reshape(AL, (1,m))
    numlayers = int(np.floor(len(caches)/4))   

    AL[AL<=0.001] = 0.001
    AL[AL>=0.999] = 0.999

    da_next = np.sum(np.divide(1-Y, 1-AL)) - np.sum(np.divide(Y, AL))
    da_next = np.divide(da_next, m)

    for i in range(numlayers, 0, -1):
#        print(da_next)    
#        da_next = np.divide(da_next, m)
#        print('Max value in da_next..'+str(np.max(da_next.flatten())))
        # Last layer has a different activation
        if (i == numlayers):
            gprime_curr = ex_activations.sigmoidDeriv(caches['Z'+str(i)])
        else:
            gprime_curr = ex_activations.reluDerivative(caches['Z'+str(i)])

        dz = np.multiply(da_next, gprime_curr)

        # Activations coming into the first layer are the inputs
        temp_al_prev = np.transpose(caches['A'+str(i-1)])

        dW = np.divide(np.matmul(dz, temp_al_prev), m)
        db = np.divide(np.sum(dz, axis=1, keepdims=True), m)
        
        da_prev = np.matmul(np.transpose(caches['W'+str(i)]), dz)
        
        da_next = da_prev
        grads['dW'+str(i)] = dW
        grads['db'+str(i)] = db
#        print('Activations of prev layer..', caches['A'+str(i-1)].shape)
#        print('Nans in dW.....'+str(dW.shape)+'...'+str(np.count_nonzero(np.isnan(dW))))    
#        print('Nans in db.....'+str(db.shape)+'...'+str(np.count_nonzero(np.isnan(db))))    

#    print(np.transpose(grads['dW'+str(numlayers)]))    
    return grads

def main():
    # Test the  function
    layers = [12288, 7, 5, 1] #  5-layer model
    
    params = ex_init_layer_weights.initialize_parameters_deep(layers)
    X = np.random.rand(layers[0],10)
    
    ex_AL, ex_caches = ex_fwd_prop.L_model_forward(X, params)
    #print(ex_AL.shape)
    #print(len(ex_caches))
    out = np.random.rand(1000,1)
    y = np.floor(np.random.rand(1000,1)+0.5)
    c = ex_compute_cost.compute_cost(out, y)
    print(c)
    grads = L_model_backward(out, y, ex_caches)
    print(len(grads))
    print(np.transpose(params['W3']))
    print(np.transpose(grads['dW3']))
#    print(grads['db'+str(j)].shape)
        

#    z = np.random.rand(3,3)-0.5
#    a = reluDerivative(z)
#    print(a)
#    gprime = np.zeros(z.shape)
#    gprime[z > 0] = 1
#    print(gprime)
    
if __name__ == "__main__":
    main()


