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
    Y = np.reshape(Y, (len(Y),1))
    AL = np.reshape(AL, (AL.shape[0] * AL.shape[1],1))
    
    a0 = AL[np.nonzero(Y==0)]
    a1 = AL[np.nonzero(Y==1)]
    
    a0[a0==1] = 0.00001
    a1[a1==0] = 0.00001
    
    cost = -np.sum(np.log(1-a0)) - np.sum(np.log(a1))
    return cost

def main():
    out = np.random.rand(1000,1)
    y = np.floor(np.random.rand(1000,1)+0.5)
    print(min(y), max(y), min(out), max(out))
    c = compute_cost(out, y)
    print(c)

if __name__ == "__main__":
    main()

