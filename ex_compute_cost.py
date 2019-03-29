#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:17:14 2019

Computes the cost from the output layer and labeled outputs
NEEDS updating to accommodate linearly varying data

@author: Naveen Kolli
"""
import numpy as np

# Logistic regression cost for now
def compute_cost(AL, Y):
    
#    cost = np.sum(np.multiply((1-Y),np.log((1-AL))) + np.multiply((Y),np.log((AL))))
    m = len(Y.flatten())
    # Condition the shapes so that output and labels align
    Y = np.reshape(Y, (1,m))
    AL = np.reshape(AL, (1,m))
    
    cost = -1/m * (np.sum(np.multiply(1-Y, np.log(1-AL))) + np.sum(np.multiply(Y, np.log(AL))))
    
    # Also return success rate
    x = (AL >= 0.5)

    success = 1 - (np.count_nonzero(Y - x)/m)
    
    return cost, success

# Logistic regression cost for now
def compute_L2_reg_cost(AL, Y, parameters, L2_lambd):
    
#    cost = np.sum(np.multiply((1-Y),np.log((1-AL))) + np.multiply((Y),np.log((AL))))
    m = len(Y.flatten())
    # Condition the shapes so that output and labels align
    Y = np.reshape(Y, (1,m))
    AL = np.reshape(AL, (1,m))

    # Cross entropy cost    
    cost = -1/m * (np.sum(np.multiply(1-Y, np.log(1-AL))) + np.sum(np.multiply(Y, np.log(AL))))
    
    # Compute the L2 reg cost component
    L2cost = 0
    for i in range(1, int(len(parameters.keys())/2)+1):
        L2cost += np.sum(np.square(parameters['W'+str(i)]))

    L2cost = (L2_lambd * L2cost)/(2*m)
    
    # Also return success rate
    x = (AL >= 0.5)

    success = 1 - (np.count_nonzero(Y - x)/m)
    
    return (cost + L2cost), success

def main():
    out = np.random.rand(1000,1)
    out[out <= 0.5] = 0.001
    out[out > 0.5] = 0.999
    
#    out = np.floor(np.random.rand(20000 ,1)+0.5)
    y = np.floor(np.random.rand(1000 ,1)+0.5)
    print(min(y), max(y), min(out), max(out))
    c = compute_cost(out, y)
    print(c)

if __name__ == "__main__":
    main()

