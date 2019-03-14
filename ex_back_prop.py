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
import ex_compute_cost
import ex_fwd_prop
import ex_init_layer_weights

def relu(a):
    rel = np.maximum(a, 0)
    return rel


def L_model_backward(AL, Y, caches):

    grads = {}
    m = len(Y.flatten())
    # Condition the shapes so that output and labels align
    Y = np.reshape(Y, (1,m))
    AL = np.reshape(AL, (1,m))
    numlayers = int(np.floor(len(caches)/3))   

    AL[AL==0] = 0.001
    AL[AL==1] = 0.999

    print('Nans in AL.....'+str(np.count_nonzero(np.isnan(AL))))    
    
    da_next = -np.sum(np.divide(1-Y, 1-AL)) + np.sum(np.divide(Y, AL))

    for i in range(numlayers, 0, -1):
#        print(da_next)    
        da_next = np.divide(da_next, m)

        gprime = np.zeros(caches['Z'+str(i)].shape)
#        gprime[caches['Z'+str(i)] > 0] = caches['Z'+str(i)][caches['Z'+str(i)] > 0]
        gprime[caches['Z'+str(i)] > 0] = 1
        
        dz = np.multiply(da_next, gprime)
        temp_al_1 = np.transpose(relu(caches['Z'+str(i-1)]))
        dW = np.divide(np.nan_to_num(np.matmul(dz, temp_al_1), False), m)
        db = np.divide(np.sum(dz, axis=1, keepdims=True), m)
        da_prev = np.matmul(np.transpose(caches['W'+str(i)]), dz)
#        print('Back prop Layer ..'+str(i)+'...dW shape..'+str(dW.shape))
        
        da_next = da_prev
        grads['dW'+str(i)] = dW
        grads['db'+str(i)] = db
        print('Nans before dW....'+str(np.count_nonzero(np.isnan(np.matmul(dz, temp_al_1)))))    

        print('Nans in dW.....'+str(dW.shape)+'...'+str(np.count_nonzero(np.isnan(dW))))    
        print('Nans in db.....'+str(db.shape)+'...'+str(np.count_nonzero(np.isnan(db))))    

        
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
    #for j in range(1, int(len(grads)/2)+1):
    #    print(grads['dW'+str(j)].shape)
    #    print(grads['db'+str(j)].shape)
        

    
    
if __name__ == "__main__":
    main()


