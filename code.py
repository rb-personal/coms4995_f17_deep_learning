#!/usr/bin/python

import numpy as np

# g(z) = 1 / (1 + exp(-z))
def sigmoid(z):
    A = 1 / (1+np.exp(-z))
    return A

# Rectified linear limit (chop off the negative portion)
def relu(z):
    A = np.maximum(0,z)
    return A

# A{l} = g{l}(w{l} * A{l-1} + b{l})
# g{l} is a non-linear function applied at layer l
# w{l} and b{l} are coefficients of layer l which transform A{l-1} linearly
def forwardPropagation(A_prv, w, b):
    # Linear Function
    z + np.matmul(W, A_prev) + b

    # Non-Linear function
    A = relu(z)

    return z
