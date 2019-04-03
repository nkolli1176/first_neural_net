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

def loadSavedParams(folder, layers_dims):

    num_layers = len(layers_dims)
    
    ### Load parameters from training
    parameters = {}
    for i in range(num_layers-1):
        parameters['W'+str(i+1)] = np.loadtxt(folder+'/Out_W'+str(i+1))
        if (len(parameters['W'+str(i+1)].shape) < 2):
            parameters['W'+str(i+1)] = np.reshape(parameters['W'+str(i+1)], (1, len(parameters['W'+str(i+1)])))
            
        parameters['b'+str(i+1)] = np.loadtxt(folder+'/Out_b'+str(i+1))
        if (len(parameters['b'+str(i+1)].shape) < 1):
            parameters['b'+str(i+1)] = np.reshape(parameters['b'+str(i+1)], (1, 1))
        else:
            parameters['b'+str(i+1)] = np.reshape(parameters['b'+str(i+1)], (len(parameters['b'+str(i+1)]), 1))            
    
    return parameters

def saveParams(folder, params):
    # Save parameters from training
    for i in range(1,int(len(params.keys())/2)+1):
        np.savetxt(folder+'/Out_W'+str(i), params['W'+str(i)])
        np.savetxt(folder+'/Out_b'+str(i), params['b'+str(i)])
            
        
def showImageData(X, Y, t_size):
    
#    # To view the images, make sure labeling is correct
    imgi = int(input('Enter an index..'))
    while (imgi != 99):
        print('Index chosen is...', imgi)
        print('Label is...', Y[imgi])
        reimg = Image.fromarray(np.reshape(X[:,imgi],(t_size)))
#        t_size = (320,320)
#        reimg = reimg.resize(t_size, Image.ANTIALIAS)
        plt.imshow(reimg)
        imgi = int(input('Enter an index..'))

def splitKfold(X, Y, j, kfolds, m):

    # K-fold xval, separate into train and x_val
    # If no kfold just use 10% as xval
    if (kfolds == 1):
        xval = int(m/10)
        x_train = X[:,0:m-xval]
        y_train = Y[0:m-xval]    
        x_xval = X[:,m-xval+1:]
        y_xval = Y[m-xval+1:]
    else:
        xval = int(m/kfolds)
        tmp_range = range(j*xval, (j+1)*xval)
        x_xval = X[:, tmp_range]
        y_xval = Y[tmp_range]
        # splits X and Y
        if (j==0):
            tmp_range = range((j+1)*xval+1, m)
            x_train = X[:, tmp_range]
            y_train = Y[tmp_range]
        else:
            tmp_range1 = range(0, j*xval)
            tmp_range2 = range((j+1)*xval+1, m)
            x_train = np.concatenate((X[:, tmp_range1], X[:, tmp_range2]), axis=1)
            y_train = np.concatenate((Y[tmp_range1], Y[tmp_range2]), axis=0)
            
#    print(x_xval.shape, y_xval.shape, x_train.shape, y_train.shape)
    return x_xval, y_xval, x_train, y_train

def L_layer_model(X, Y, parameters, layers_dims, kfolds_num, numbatches, learning_rate = 0.001, num_iterations = 3000, print_cost=1, localfolder):#lr was 0.009
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
    if (len(parameters) == 0):
        parameters = ex_init_layer_weights.initialize_parameters_deep(layers_dims)

    newparams = copy.deepcopy(parameters)
    
    # Separate ino Training and Cross validation sets
    m = X.shape[1]
        
    starttime = time.time()
    # Loop (gradient descent)
    for i in range(0, int(num_iterations/kfolds_num)): 

        for j in range(0, kfolds_num):
            
            X_xval, Y_xval, X_train, Y_train = splitKfold(X, Y, j, kfolds_num, m)
            # Batch training, numbatches is 1 if no batch separation
            batch_size = int(X_train.shape[1]/numbatches)
        
            for k in range(0, numbatches):
                
                X_batch = X_train[:,(k*batch_size):((k+1)*batch_size-1)]
                Y_batch = Y_train[(k*batch_size):((k+1)*batch_size-1)]
    
                # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.            
                AL, caches = ex_fwd_prop.L_model_forward(X_batch, newparams)
                
                # Compute cost
                cost, train_success = ex_compute_cost.compute_cost(AL, Y_batch)
                # Compute cost with L2 reg
                # cost, train_success = ex_compute_cost.compute_L2_reg_cost(AL, Y_batch, newparams, L2_lambd)
            
                # Backward propagation
                grads = ex_back_prop.L_model_backward(AL, Y_batch, caches)
                # Backward propagation with L2 reg
                # grads = ex_back_prop.L_model_backward_L2_reg(AL, Y_batch, caches, L2_lambd)
         
                # Update parameters.
                newparams = ex_update_parameters.update_parameters(newparams, grads, learning_rate)
                
        # Print the cost every nth training cycle
        if print_cost and i % 10 == 0:
            endtime = time.time()
            print ("Cost after iteration %i: %f, Time: %f" %(i, cost, (endtime-starttime)))
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
            # Optional save parameters every nth iteration for future use
            saveParams(localfolder, newparams)

            
    return parameters, newparams, train_success

def train_data(localfolder, layers_dims, kfolds, numbatches, learning_rate, num_epochs, print_cost, use_saved_params):
    
    # Load dataset 
    X_train = np.loadtxt(localfolder+'/X_K_mono_train.dat')
    Y_train = np.loadtxt(localfolder+'/Y_K_mono_train.dat')
    
    # Number of examples
    m = X_train.shape[1]
    print('Data loaded..' + str(time.ctime()) + '...' + str(X_train.shape))
    
    # Shuffle the data
    marr = np.arange(m-1)
    np.random.shuffle(marr)
    X_train = X_train[:,marr]
    Y_train = Y_train[marr]
    
    X_train = X_train/255
#    X_train[X_train <= 0.4] = 0
#    X_train[X_train > 0.4] = 1
    
    print('Layer dims...'+str(layers_dims) + '...' + str(time.ctime()))
    parameters = {}
    if (use_saved_params):
        parameters = loadSavedParams(localfolder, layers_dims)

    params, newparams, train_success = L_layer_model(X_train, Y_train, parameters, layers_dims, kfolds, numbatches, learning_rate, num_epochs, print_cost, localfolder)
    print('Optimization done..'+str(train_success))
    
    saveParams(localfolder, newparams)

    

def test_data(localfolder, layers_dims, t_size):
    
    # Load test data
    X_test = np.loadtxt(localfolder+'/X_K_mono_test.dat')
    Y_test = np.loadtxt(localfolder+'/Y_K_mono_test.dat')
    
    # Number of examples
    m = X_test.shape[1]
    
    # Shuffle the data
    marr = np.arange(m-1)
    np.random.shuffle(marr)
    X_test = X_test[:,marr]
    Y_test = Y_test[marr]
    
    X_test = X_test/255
#    X_train[X_train <= 0.4] = 0
#    X_train[X_train > 0.4] = 1
    
    parameters = loadSavedParams(localfolder, layers_dims)
    
    ### Run forward prop to get the output activations    
    AL, caches = ex_fwd_prop.L_model_forward(X_test, parameters)
    
    ### Convert AL to binary calls
    calls = (AL >= 0.5)
    
    ## Calculate success percentage
    success = 1 - np.count_nonzero(Y_test - calls)/m

    # Show images from test data and the classification results    
    showImageData(X_test, calls, (t_size))    
    
    return success

def main():
    
    localfolder = 'dim_128_mono'
    starttime = time.time()
    print(time.ctime())

    # Define input feature length - number of pixels in the images here
    img_h = 128
    img_w = 128
    img_d = 1

    dim_1 = img_h * img_w * img_d
    layers_dims = [dim_1, 25, 13, 5, 1] #  5-layer model

    ### Train data
    kfolds = 10
    numbatches = 20
    learning_rate = 0.001
    num_epochs = 10000
    print_cost = 1
    use_saved_params = 1
    train_data(localfolder, layers_dims, kfolds, numbatches, learning_rate, num_epochs, print_cost, use_saved_params)
    
    ### Test data
    success = test_data(localfolder, layers_dims, (img_w, img_h))

    endtime = time.time()
    print('Time {0}, num_epochs {1}, success {2}'.format((endtime-starttime), num_epochs, success))
    
if __name__ == "__main__":
    main()
 
    
