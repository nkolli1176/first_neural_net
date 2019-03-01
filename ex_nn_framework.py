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
layers_dims = [12288, 20, 13, 7, 5, 1] #  5-layer model

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
    caches = {}
    # Gotta save the inputs as the first activations
    caches["Z0"]=x0

    for i in range(1, int(len(parameters.keys())/2)+1):
        print('Fwd prop Layer..'+str(i)+'...shape'+str(parameters['W'+str(i)].shape))
        zl = np.matmul(parameters['W'+str(i)], x0) + parameters['b'+str(i)]
        x0 = relu(zl)
        caches["Z"+str(i)]=x0
        caches['W'+str(i)]=parameters['W'+str(i)]
        caches['b'+str(i)]=parameters['b'+str(i)]        

    # Output layer Activation is sigmoid        
    AL = sigmoid(x0)    
     
    return AL, caches

# Cost function: Logistic regression for now
def compute_cost(AL, Y):
    
#    cost = np.sum(np.multiply((1-Y),np.log((1-AL))) + np.multiply((Y),np.log((AL))))
    a0 = AL[np.nonzero(Y)]
    a1 = AL[np.nonzero(Y==1)]
    cost = -np.sum(np.log(1-a0) + np.log(a1))
    return cost

# Back prop
def L_model_backward(AL, Y, caches):

    grads = {}
    m = Y.shape[0]
    numlayers = int(np.floor(len(caches)/3))   
#    print(m)
    da_next = np.sum(-np.divide(-Y[np.nonzero(Y==1)], AL[np.nonzero(Y==1)])) + np.sum(np.divide((1-Y[np.nonzero(Y==0)]), (1-AL[np.nonzero(Y==0)])))
    da_next = da_next/m
#    print(numlayers)
    for i in range(numlayers, 0, -1):
        gprime = np.zeros(caches["Z"+str(i)].shape)
        gprime[np.nonzero(caches["Z"+str(i)])] = caches["Z"+str(i)][np.nonzero(caches["Z"+str(i)])]

        dz = np.multiply(da_next, gprime)
        dW = 1/m * np.matmul(dz, np.transpose(relu(caches["Z"+str(i-1)])))
        db = 1/m * np.sum(dz, axis=1, keepdims=True)
        da_prev = np.matmul(np.transpose(caches["W"+str(i)]), dz)
        print("Back prop Layer .."+str(i)+"...dW shape.."+str(dW.shape))
        
        da_next = da_prev
        grads["dW"+str(i)] = dW
        grads["db"+str(i)] = db

        
    return grads

# Update parameters, use gradients and learning_rate
def update_parameters(parameters, grads, learning_rate):

    for i in range(1, int(len(parameters.keys())/2)+1):
        print('Updating Layer..'+str(i))
        parameters['W'+str(i)] = parameters['W'+str(i)] - (learning_rate * grads['dW'+str(i)])
        parameters['b'+str(i)] = parameters['b'+str(i)] - (learning_rate * grads['db'+str(i)])

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

 
# Next steps
#    1. Find dataset, split into test, validation and training sets
#    2. Run training set, get new params
#    3. Run hyper-param optimization using validation
#    4. Run on test set with final params
    