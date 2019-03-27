#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:13:42 2019

Framework for NN, fully connected layers followed by linear activation for output
Output activation should be a linear function
Functions are written and tested separately where possible

@author: Naveen Kolli
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import copy
# functions to implement the model
import ex_fwd_prop
import ex_init_layer_weights
import ex_update_parameters
import ex_back_prop
import ex_compute_cost



def L_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 3000, print_cost=1):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    # NK: keep this for now, for testing the model
#    np.random.seed(1)
    costs = []                         # keep track of cost
    train_successes = []
    xval_successes = []
    # Parameters initialization
    parameters = ex_init_layer_weights.initialize_parameters_deep(layers_dims)
    newparams = copy.deepcopy(parameters)
    
    # Separate ino Training and Cross validation sets
    m = X.shape[1]
    xval = int(m/5)
    X_train = X[:,0:m-xval]
    Y_train = Y[0:m-xval]

    X_xval = X[:,m-xval+1:]
    Y_xval = Y[m-xval+1:]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations): 

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = ex_fwd_prop.L_model_forward(X_train, newparams)
        
        # Compute cost.
        cost, train_success = ex_compute_cost.compute_cost(AL, Y_train)
    
        # Backward propagation.
        grads = ex_back_prop.L_model_backward(AL, Y_train, caches)
 
        # Update parameters.
        newparams = ex_update_parameters.update_parameters(newparams, grads, learning_rate)
                
        # Print the cost every nth training cycle
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            train_successes.append(train_success)
            # Run xval set and compute cost.
            AL_xval, xval = ex_fwd_prop.L_model_forward(X_xval, newparams)        
            xval_cost, xval_success = ex_compute_cost.compute_cost(AL_xval, Y_xval)
            xval_successes.append(xval_success)
            # plot the cost
            plt.plot(np.squeeze(train_successes))
            plt.plot(np.squeeze(xval_successes))            
            plt.ylabel('Success %')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.draw()
            plt.pause(1)

            
    return parameters, newparams, train_success

def train_data(localfolder, layers_dims, learning_rate, num_epochs, print_cost):
    
    # Load dataset 
    X_train = np.loadtxt(localfolder+'/X_train.dat')
    Y_train = np.loadtxt(localfolder+'/Y_train.dat')
    
    # Number of examples
    m = X_train.shape[1]
    
    # Shuffle the data
    marr = np.arange(m-1)
    np.random.shuffle(marr)
    X_train = X_train[:,marr]
    Y_train = Y_train[marr]

#    # To view the images, make sure labeling is correct
#    imgi = int(input('Enter an index..'))
#    while (imgi != 99):
#        print('Index chosen is...', imgi)
#        print('Label is...', Y_train[imgi])
#        reimg = Image.fromarray(np.reshape(X_train[:,imgi],(32,32)))
#        t_size = (320,320)
#        reimg = reimg.resize(t_size, Image.ANTIALIAS)
#        reimg.show()
#        imgi = int(input('Enter an index..'))
    
    X_train = X_train/255
#    X_train[X_train <= 0.4] = 0
#    X_train[X_train > 0.4] = 1
    
    print('Layer dims...'+str(layers_dims))

    params, newparams, train_success = L_layer_model(X_train, Y_train, layers_dims, learning_rate, num_epochs, print_cost)
    print('Optimization done..'+str(train_success))
    
    for i in range(1,int(len(newparams.keys())/2)+1):
        np.savetxt(localfolder+'/Out_W'+str(i), newparams['W'+str(i)])
        np.savetxt(localfolder+'/Out_b'+str(i), newparams['b'+str(i)])

    
#    print(np.max(newparams['W4'] - params['W4']))
#    print(np.max(newparams['W3'] - params['W3']))
#    print(np.max(newparams['W2'] - params['W2']))
#    print(np.max(newparams['W1'] - params['W1']))

def test_data(localfolder, layers_dims):
    
    # Load test data
    X_test = np.loadtxt(localfolder+'/X_test.dat')
    Y_test = np.loadtxt(localfolder+'/Y_test.dat')
    
    # Number of examples
    m = X_test.shape[1]
    
    # Shuffle the data
    marr = np.arange(m-1)
    np.random.shuffle(marr)
    X_test = X_test[:,marr]
    Y_test = Y_test[marr]

#    # To view the images, make sure labeling is correct
#    imgi = int(input('Enter an index..'))
#    while (imgi != 99):
#        print('Index chosen is...', imgi)
#        print('Label is...', Y_train[imgi])
#        reimg = Image.fromarray(np.reshape(X_train[:,imgi],(32,32)))
#        t_size = (320,320)
#        reimg = reimg.resize(t_size, Image.ANTIALIAS)
#        reimg.show()
#        imgi = int(input('Enter an index..'))
    
    X_test = X_test/255
#    X_train[X_train <= 0.4] = 0
#    X_train[X_train > 0.4] = 1
    
    num_layers = len(layers_dims)

    ### Load parameters from training
    parameters = {}
    for i in range(num_layers-1):
        parameters['W'+str(i+1)] = np.loadtxt(localfolder+'/Out_W'+str(i+1))
        if (len(parameters['W'+str(i+1)].shape) < 2):
            parameters['W'+str(i+1)] = np.reshape(parameters['W'+str(i+1)], (1, len(parameters['W'+str(i+1)])))
            
        parameters['b'+str(i+1)] = np.loadtxt(localfolder+'/Out_b'+str(i+1))
        if (len(parameters['b'+str(i+1)].shape) < 1):
            parameters['b'+str(i+1)] = np.reshape(parameters['b'+str(i+1)], (1, 1))
        else:
            parameters['b'+str(i+1)] = np.reshape(parameters['b'+str(i+1)], (len(parameters['b'+str(i+1)]), 1))            

    ### Run forward prop to get the output activations    
    AL, caches = ex_fwd_prop.L_model_forward(X_test, parameters)
#    print('AL max is ...', np.max(AL))
#    print('AL min is ...', np.min(AL))
#    print('AL avg is ...', np.mean(AL))
#    print('AL median is ...', np.median(AL))
#    print(AL.shape, Y_test.shape)
#    print((Y_test - AL).shape)
#    print(np.count_nonzero(AL < 0.3))
    
    ### Convert AL to binary calls
    AL[AL < 0.5] = 0
    AL[AL >= 0.5] = 1

    ## Calculate success percentage
    success = 1 - np.count_nonzero(Y_test - AL)/m
    
    return success

def main():
    
    localfolder = 'Data_Kag'
    starttime = time.time()
    print(time.ctime())

    # Define input feature length - number of pixels in the images here    
    dim_1 = 12288
    layers_dims = [dim_1, 50, 13, 5, 1] #  5-layer model

    ### To train data
    learning_rate = 0.005
    num_epochs = 15000
    print_cost = 1
    train_data(localfolder, layers_dims, learning_rate, num_epochs, print_cost)
    
    ### Test data
    success = test_data(localfolder, layers_dims)
#    print('Success ratio is..', success)
    endtime = time.time()
    print('Time {0}, num_epochs {1}, success {2}'.format((endtime-starttime), num_epochs, success))
    # 0.64 best score with 0.01 lr, 10k epoch
    
if __name__ == "__main__":
    main()
 
# Next steps
#    1. Find dataset, split into test, validation and training sets
#    2. Run training set, get new params
#    3. Run hyper-param optimization using validation
#    4. Run on test set with final params
    