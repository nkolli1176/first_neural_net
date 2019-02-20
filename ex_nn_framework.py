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
import matplotlib as plt

### EXAMPLE layer dimensions ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

## Initialize all weights for all layers
def initialize_parameters_deep(layers_dims):

    num_layers = len(layers_dims)
    parameters = {}
    
    # Normal rand initialization of all weights
    for i in range(num_layers-1):
        print(i)
        parameters['W'+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i])
        parameters['b'+str(i+1)] = np.random.randn(layers_dims[i+1], )
    
    return parameters

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
def L_model_forward(X, parameters):
    x0 = X
    caches = []
    
    for i in range(1, int(len(parameters.keys())/2)+1):
        print('Processing Layer..'+str(i))
        al = np.matmul(parameters['W'+str(i)], x0) + parameters['b'+str(i)]
        x0 = relu(al)
        
        caches.append(x0)
        
    AL = sigmoid(x0)    
    caches.append(AL)
    # Output layer here is sigmoid
    
    return AL, caches

# Cost function: Logistic regression for now
def compute_cost(AL, Y):
    
#    cost = np.sum(np.multiply((1-Y),np.log((1-AL))) + np.multiply((Y),np.log((AL))))
    a0 = AL[np.nonzero(Y)]
    a1 = AL[np.nonzero(Y==1)]
    cost = -np.sum(np.log(1-a0) + np.log(a1))
    return cost

def L_model_backward(AL, Y, caches):
    return grads

def update_parameters(parameters, grads, learning_rate):
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
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
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###
    
    # Loop (gradient descent)
    for i in range(0, num_iterations): 

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

 
