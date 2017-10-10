#!/usr/bin/python
import numpy as np

class NeuralNetwork(object):

    def __init__(self, layer_dimensions, drop_prob=0.0, reg_lambda=0.0):
        np.random.seed(1)

        self.parameters = {}
        self.parameters["batch_index"] = 0
        self.num_layers = len(layer_dimensions)
        self.drop_prob = drop_prob
        self.reg_lambda = reg_lambda

        for l in range(1, self.num_layers-1):
            self.parameters["W" + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1])
            self.parameters["b" + str(l)] = np.zeros((layer_dimensions[l], 1))

    def affineForward(self, A, W, b):
        assert(W.shape[1] == A.shape[0])
        cache = (A, W, b)
        return W.dot(A)+b, cache

    def activationForward(self, A, activation="relu"):
        if activation=="relu":
            return relu(X)

    def relu(self, X):
        return np.maximum(0, X), X

    def dropout(self, A, prob):
        M = None
        return A, M

    def forwardPropagation(self, X):
        AL = None
        cache = []

        AL = X
        for l in range (1, self.num_layers-1):
            W = self.parameters["W"+str(l)]
            b = self.parameters["b"+str(l)]
            AL, cacheL_linear = affineForward(self, AL, W, b)
            AL, cacheL_nonlinear = activationForward(AL, "relu")
            cache.append((cacheL_linear, cacheL_nonlinear))

        return AL, cache

    def costFunction(self, AL, y):
        cost = (-1/AL.shape[1]) * np.sum(np.log(AL)*y + np.log(1-AL)*(1-y))
        cost = np.squeeze(cost)

        #if self.reg_lambda > 0:
        # add regularization

        dAL = - (np.divide(y, AL) - np.divide(1-y, 1-AL))
        return cost, dAL

    def affineBackward(self, dA_prev, cache):
        A, W, b = cache
        dA, dW, db = None, None, None

        dA = np.dot(W.T,dA_prev)
        dW = (1/A.shape[1]) * dA_prev.dot(A.T)
        db = (1/A.shape[1]) * np.sum(dA_prev, axis=1, keepdims=True)

        return dA, dW, db

    def relu_derivative(self, dx, cached_x):
        dx[x<=0] = 0
        return dx

    def activationBackward(self, dA, cache, activation="relu"):
        if activation=="relu":
            return relu_derivative(dA, cache)

    def dropout_backward(self, dA, cache):
        return dA

    def backPropagation(self, dAL, Y, cache):
        gradients = {}

        for l in reversed(range(1, self.num_layers-1)):
            current_cache = caches[l]
            dAL_nonlinear = activationBackward(dAL, current_cache[1], "relu")
            dAL, dW, db = affineBackward(dAL_nonlinear, current_cache[0])
            gradients["dW"+str(l)] = dW
            gradients["db"+str(l)] = db

            #if self.drop_prob > 0:
            #call dropout_backward

        #if self.reg_lambda > 0:
        # add gradients from L2 regularization to each dW

        return gradients

    def updateParameters(self, gradients, alpha):
        for l in range(1, self.num_layers-1):
            dW = gradients["dW"+str(l)]
            db = gradients["db"+str(l)]

            W_idx = "W"+str(l)
            b_idx = "b"+str(l)
            assert(dW.shape == self.parameters[W_idx])
            assert(db.shape == self.parameters[b_idx])

            self.parameters[W_idx] = self.parameters[W_idx] - alpha*dW
            self.parameters[b_idx] = self.parameters[b_idx] - alpha*db

    def train(self, X, y, iters=1000, alpha=0.0001, batch_size=100, print_every=100):
        for i in range(0, iters):
            X_batch, y_batch = get_batch(self, X, y, batch_size)
            y_hat, cache = forwardPropagation(self, X_batch)
            cost, dAL = costFunction(y_hat, y_batch)
            gradients = backPropagation(self, dAL, y_batch, cache)
            updateParameters(self, gradients, alpha)

            if i % print_every == 0:
                print("Cost: ", cost)

    def predict(self, X):
        y_pred, cache = forwardPropagation(X)
        return y_pred

    def get_batch(self, X, y, batch_size):
        bi = self.parameters["batch_index"]
        X_batch = X[:,bi:bi+batch_size]
        y_batch = y[bi:bi+batch_size]
        self.parameters["batch_index"] = self.parameters["batch_index"]+batch_size
        assert(y_batch.shape[0] == X_batch.shape[1])
        return X_batch, y_batch


layer_dims = (3000, 5, 10)
nn = NeuralNetwork(layer_dims)
