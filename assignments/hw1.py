#!/usr/bin/python

class NeuralNetwork(object):

# Abstraction of neural network.
# Stores parameters, activations, cached values.
Provides necessary functions for training and prediction.
"""
def __init__(self, layer_dimensions, drop_prob=0.0, reg_lambda=0.0):
    """
    Initializes the weights and biases for each layer
    :param layer_dimensions: (list) number of nodes in each layer
    :param drop_prob: drop probability for dropout layers. Only required in part 2 of the assignment
    :param reg_lambda: regularization parameter. Only required in part 2 of the assignment
    """
    np.random.seed(1)

    self.parameters = {}
    self.parameters["batch_index"] = 0
    self.num_layers = len(layer_dimensions)
    self.drop_prob = drop_prob
    self.reg_lambda = reg_lambda

    # init parameters
    for l in range(1, self.num_layers-1):
        self.parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1])
        self.parameters['b' + str(l)] = np.zeros(layer_dims[l], 1)

def affineForward(self, A, W, b):
    """
    Forward pass for the affine layer.
    :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
    the number of samples
    :returns: the affine product WA + b, along with the cache required for the backward pass
    """
    assert(W.shape[1] == A.shape[0])
    cache = (A, W, b)
    return W.dot(A)+b, cache

def activationForward(self, A, activation="relu"):
    """
    Common interface to access all activation functions.
    :param A: input to the activation function
    :param prob: activation funciton to apply to A. Just "relu" for this assignment.
    :returns: activation(A)
    """
    if activation=="relu":
        return relu(X)

def relu(self, X):
    return np.maximum(0, X), X

def dropout(self, A, prob):
    """
    :param A:
    :param prob: drop prob
    :returns: tuple (A, M)
        WHERE
        A is matrix after applying dropout
        M is dropout mask, used in the backward pass
    """

    return A, M

def forwardPropagation(self, X):
    """
    Runs an input X through the neural network to compute activations
    for all layers. Returns the output computed at the last layer along
    with the cache required for backpropagation.
    :returns: (tuple) AL, cache
        WHERE
        AL is activation of last layer
        cache is cached values for each layer that
                 are needed in further steps
    """
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
    """
    :param AL: Activation of last layer, shape (num_classes, S)
    :param y: labels, shape (S)
    :param alpha: regularization parameter
    :returns cost, dAL: A scalar denoting cost and the gradient of cost
    """
    # compute loss
    cost = (-1/AL.shape[1]) * np.sum(np.log(AL)*y + np.log(1-AL)*(1-y))
    cost = np.squeeze(cost)

    if self.reg_lambda > 0:
        # add regularization


    # gradient of cost
    dAL = - (np.divide(y, AL) - np.divide(1-y, 1-AL))
    return cost, dAL

def affineBackward(self, dA_prev, cache):
    """
    Backward pass for the affine layer.
    :param dA_prev: gradient from the next layer.
    :param cache: cache returned in affineForward
    :returns dA: gradient on the input to this layer
             dW: gradient on the weights
             db: gradient on the bias
    """
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
    """
    Interface to call backward on activation functions.
    In this case, it's just relu.
    """
    if activation=="relu":
        return relu_derivative(dA, cache)

def dropout_backward(self, dA, cache):

    return dA

def backPropagation(self, dAL, Y, cache):
    """
    Run backpropagation to compute gradients on all paramters in the model
    :param dAL: gradient on the last layer of the network. Returned by the cost function.
    :param Y: labels
    :param cache: cached values during forwardprop
    :returns gradients: dW and db for each weight/bias
    """
    gradients = {}

    for l in reversed(range(1, self.num_layers-1)):
        current_cache = caches[l]
        dAL_nonlinear = activationBackward(dAL, current_cache[1], "relu")
        dAL, dW, db = affineBackward(dAL_nonlinear, current_cache[0])
        gradients["dW"+str(l)] = dW
        gradients["db"+str(l)] = db

        if self.drop_prob > 0:
            #call dropout_backward


    if self.reg_lambda > 0:
        # add gradients from L2 regularization to each dW

    return gradients


def updateParameters(self, gradients, alpha):
    """
    :param gradients: gradients for each weight/bias
    :param alpha: step size for gradient descent
    """
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
    """
    :param X: input samples, each column is a sample
    :param y: labels for input samples, y.shape[0] must equal X.shape[1]
    :param iters: number of training iterations
    :param alpha: step size for gradient descent
    :param batch_size: number of samples in a minibatch
    :param print_every: no. of iterations to print debug info after
    """

    print(X.shape)
    print(y.shape)
    print(y)
    exit()

    for i in range(0, iters):
        # get minibatch
        X_batch, y_batch = get_batch(self, X, y, batch_size)

        # forward prop
        y_hat, cache = forwardPropagation(self, X_batch)

        # compute loss
        assert(y_hat.shape[0] == y_batch.shape[0])
        cost, dAL = costFunction(y_hat, y_batch)

        # compute gradients
        gradients = backPropagation(self, dAL, y_batch, cache)

        # update weights and biases based on gradient
        updateParameters(self, gradients, alpha)

        if i % print_every == 0:
            # print cost, train and validation set accuracies
            print("Cost: ", cost)


def predict(self, X):
    """
    Make predictions for each sample
    """
    y_pred, cache = forwardPropagation(X)
    return y_pred

def get_batch(self, X, y, batch_size):
    """
    Return minibatch of samples and labels

    :param X, y: samples and corresponding labels
    :param batch_size: minibatch size
    :returns: (tuple) X_batch, y_batch
    """
    bi = self.parameters["batch_index"]
    X_batch = X[:,bi:bi+batch_size]
    y_batch = y[bi:bi+batch_size]
    self.parameters["batch_index"] = self.parameters["batch_index"]+batch_size
    assert(y_batch.shape[0] == X_batch.shape[1])
    return X_batch, y_batch
