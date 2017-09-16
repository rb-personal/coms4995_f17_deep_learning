#!/usr/bin/python

import numpy as np

# Cost Function J
def J(theta, x):
    return theta * x

def JDeriv(theta, x):
    return theta

def forwardPropagation(theta, x):
    return J(theta, x)

def backwardPropagation(theta, x):
    return JDeriv(theta, x)

def norm(x):
    return np.linalg.norm(x)

def gradientCheck(theta, x):
    eps = 1e-7
    J_pos = J(theta + eps, x)
    J_neg = J(theta - eps, x)
    numerical_deriv = (J_pos - J_neg) / (2*eps)
    print("numerical deriv = ", numerical_deriv)

    analytic_deriv = JDeriv(theta, x)
    print("analytic_deriv = ", analytic_deriv)

    error = norm(analytic_deriv - numerical_deriv) / (norm(numerical_deriv) + norm(analytic_deriv))
    return error

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

theta = 10
x = 3
error = gradientCheck(theta, x)
print("error = ", error)
