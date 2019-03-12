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
# functions to implement the model
import ex_init_layer_weights
import ex_update_parameters
import ex_fwd_prop
import ex_back_prop
import ex_compute_cost

## Sigmoid Activation function
def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig

## RELU Activation function
def relu(a):
    rel = np.maximum(a, 0)
    return rel

# Perform forward propagation
    # NEEDS UPDATE: last activation, should be linear and multi-hot(!)


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 20, print_cost=1):#lr was 0.009
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
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = ex_init_layer_weights.initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations): 

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = ex_fwd_prop.L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = ex_compute_cost.compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = ex_back_prop.L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = ex_update_parameters.update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 1 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def main():
    # Load dataset 
    X_train = np.loadtxt('Data/X_train.dat')
    Y_train = np.loadtxt('Data/Y_train.dat')
 
#    X_train = X_train > 0.2
    
    dim_1 = X_train.shape[0]
    ### Layer dimensions ###
    layers_dims = [dim_1, 20, 13, 7, 5, 1] #  5-layer model
    print('Layer dims...'+str(layers_dims))

    newparams = L_layer_model(X_train, Y_train, layers_dims)
    print('Optimization done..'+str(len(newparams)))
    
    for i in range(1,int(len(newparams.keys())/2)+1):
        np.savetxt('Data/Out_W'+str(i), newparams['W'+str(i)])
        np.savetxt('Data/Out_b'+str(i), newparams['b'+str(i)])

    print('Program done..')
    
if __name__ == "__main__":
    main()
 
# Next steps
#    1. Find dataset, split into test, validation and training sets
#    2. Run training set, get new params
#    3. Run hyper-param optimization using validation
#    4. Run on test set with final params
    