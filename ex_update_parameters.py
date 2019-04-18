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
import matplotlib.pyplot as plt
import copy

def update_params_Adam(mom_prev, rms_prev, beta_mom, beta_rms, learning_rate, epsilon, niter, parameters, grads):

    m = 1
    if (len(mom_prev)==0):
        for j in range(1, int(len(parameters.keys())/2)+1):
            mom_prev['VdW'+str(j)] = np.zeros((parameters['W'+str(j)].shape))
            mom_prev['Vdb'+str(j)] = np.zeros((parameters['b'+str(j)].shape))
            rms_prev['SdW'+str(j)] = np.zeros((parameters['W'+str(j)].shape))
            rms_prev['Sdb'+str(j)] = np.zeros((parameters['b'+str(j)].shape))

    for i in range(1, int(len(parameters.keys())/2)+1):        
#        print(np.max(grads['dW'+str(i)]/m))
        mom_prev['VdW'+str(i)] = (beta_mom * mom_prev['VdW'+str(i)]) + ((1 - beta_mom) * grads['dW'+str(i)]/m)
        tmp_VdW = np.divide(mom_prev['VdW'+str(i)], (1 - np.power(beta_mom, niter)))
        mom_prev['Vdb'+str(i)] = (beta_mom * mom_prev['Vdb'+str(i)]) + ((1 - beta_mom) * grads['db'+str(i)]/m)
        tmp_Vdb = np.divide(mom_prev['Vdb'+str(i)], (1 - np.power(beta_mom, niter)))

        rms_prev['SdW'+str(i)] = (beta_rms * rms_prev['SdW'+str(i)]) + ((1 - beta_rms) * np.square(grads['dW'+str(i)]/m))
        tmp_SdW = np.divide(rms_prev['SdW'+str(i)], (1 - np.power(beta_rms, niter)))
        rms_prev['Sdb'+str(i)] = (beta_rms * rms_prev['Sdb'+str(i)]) + ((1 - beta_rms) * np.square(grads['db'+str(i)]/m))
        tmp_Sdb = np.divide(rms_prev['Sdb'+str(i)], (1 - np.power(beta_rms, niter)))                

#        parameters['W'+str(i)] = parameters['W'+str(i)] - (learning_rate * np.divide(tmp_VdW, np.sqrt(tmp_SdW) + epsilon))
#        parameters['b'+str(i)] = parameters['b'+str(i)] - (learning_rate * np.divide(tmp_Vdb, np.sqrt(tmp_Sdb) + epsilon))
#        parameters['W'+str(i)] = parameters['W'+str(i)] - (learning_rate * tmp_VdW)
#        parameters['b'+str(i)] = parameters['b'+str(i)] - (learning_rate * tmp_Vdb)
        parameters['W'+str(i)] = parameters['W'+str(i)] - (learning_rate * np.divide(grads['dW'+str(i)], (np.sqrt(tmp_SdW) + epsilon)))
        parameters['b'+str(i)] = parameters['b'+str(i)] - (learning_rate * np.divide(grads['db'+str(i)], (np.sqrt(tmp_Sdb) + epsilon)))
        
    return mom_prev, rms_prev, parameters

def update_parameters(parameters, grads, learning_rate):

    for i in range(1, int(len(parameters.keys())/2)+1):
#        print('Updating Layer..'+str(i))
        parameters['W'+str(i)] = parameters['W'+str(i)] - (learning_rate * grads['dW'+str(i)])
        parameters['b'+str(i)] = parameters['b'+str(i)] - (learning_rate * grads['db'+str(i)])

    return parameters


def main():
    # Test the  function
    np.random.seed(1)
    layers = [20, 13, 7, 5, 1] #  4-layer model
    m = 1000
    params = ex_init_layer_weights.initialize_parameters_deep(layers)
    X = np.random.rand(layers[0],m)
    
    ex_AL, ex_caches = ex_fwd_prop.L_model_forward(X, params)
#    print(ex_caches['Z4'])    
    out = np.random.rand(m,1)
    y = np.floor(np.random.rand(m,1)+0.5)
    grads = ex_back_prop.L_model_backward(out, y, ex_caches)
    print(grads['db4'])
    print(grads['db1'].shape, grads['db2'].shape, grads['db3'].shape, grads['db4'].shape)

#    newparams = update_parameters(params, grads, 0.008)
#    print(len(newparams))
#    print(params['b3'] - newparams['b3'])
#    print(newparams['b4'])

    newparams = copy.deepcopy(params)
    mom_prev = {}
    rms_prev = {}
    maxes = []
    newmaxes = []    
    n = 3
    for i in range(1,100):
        mom_prev, rms_prev, newparams = update_params_Adam(mom_prev, rms_prev, 0.5, 0, 0.01, 1e-8, i, newparams, grads)
        newmaxes.append((newparams['b3'][n]))
        
        params = update_parameters(params, grads, 0.01)
        maxes.append((params['b3'][n]))


    plt.plot(np.squeeze(maxes), 'k')            
    plt.plot(np.squeeze(newmaxes), 'b')                
    plt.ylabel('Success %')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = 0.01")
    plt.draw()


    
if __name__ == "__main__":
    main()

