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
    a0 = AL[np.nonzero(Y)]
    a1 = AL[np.nonzero(Y==1)]
    cost = -np.sum(np.log(1-a0) + np.log(a1))
    return cost

out = np.random.rand(1000,1)
y = np.floor(np.random.rand(1000,1)+0.5)
#print(min(y), max(y), min(out), max(out))
c = compute_cost(out, y)
print(c)
