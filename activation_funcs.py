import numpy as np

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def softmax(Z):
    return np.exp(Z)/(np.sum(np.exp(Z), axis = 0))

def sigmoid_deriv(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def softmax_deriv(Z):
    return 1.0/(1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    deriv = Z
    deriv[deriv<=0] = 0
    deriv[deriv>0] = 1
    return deriv

def tanh_deriv(Z):
    return (1 - np.tanh(Z) ** 2)