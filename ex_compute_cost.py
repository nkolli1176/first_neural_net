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
    
    AL[AL==0] = 0.001
    AL[AL==1] = 0.999
    
    cost = -1/m * (np.sum(np.multiply(1-Y, np.log(1-AL))) + np.sum(np.multiply(Y, np.log(AL))))
    
    return cost

def main():
    out = np.random.rand(1000,1)
    y = np.floor(np.random.rand(1000,1)+0.5)
    print(min(y), max(y), min(out), max(out))
    c = compute_cost(out, y)
    print(c)

if __name__ == "__main__":
    main()

