#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:39:39 2019
Useful math functions
Activations commonly used and their derivatives
@author: Naveen Kolli
"""

import numpy as np


def sigmoid(z):
    sig = 1/(1 + np.exp(-z))
    return sig

def relu(a):
    rel = np.maximum(a, 0)
    return rel

def reluDerivative(x):
   x[x<=0] = 0
   x[x>0] = 1
   return x

def sigmoidDeriv(z):
    sd = sigmoid(z)*(1 - sigmoid(z))
    return sd


def main():
    
    A = np.random.rand(3,4)
    print(A)
    sig = sigmoid(A)
    print(sig)
    sd = sigmoidDeriv(A)
    print(sd)
    r = relu(A-0.5)
    print(r)
    rd = reluDerivative(A-0.5)
    print(rd)


if __name__ == "__main__":
    main()

