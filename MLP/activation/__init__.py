import numpy as np
from enum import Enum


class Activation(Enum):
    LINEAR = 0
    SIGMOID = 1
    TANH = 2
    RELU = 3
    SOFTMAX = 4

def sigmoid(y):
    return np.vectorize(lambda x: 1/(1+np.e**(-x)))(y)

def dsigmoid(y):
    return np.vectorize(lambda x: np.e**(-x)/(1+np.e**(-x))**2)(y)

def linear(y):
    return y

def dlinear(y):
    return np.vectorize(lambda x: 1)(y)

def tanh(y):
    return np.tanh(y)

def dtanh(y):
    return 1/(np.cosh(y)**2)

def ReLU(y):
    return np.vectorize(lambda x: max(0,x))(y)

def dReLU(y):
    return np.where(y==0,np.random.uniform(),y>0)

def softmax(z):
    e = np.exp(z-np.max(z))
    s = np.sum(e, axis=1, keepdims=True)
    return e/s

def dsoftmax(da, z):
    m, n = z.shape
    p = softmax(z)
    tensor1 = np.einsum('ij,ik->ijk', p, p)
    tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))
    dSoftmax = tensor2 - tensor1
    dz = np.einsum('ijk,ik->ij', dSoftmax, da)
    return dz

activation_functions_gradients = {
    Activation.LINEAR: (linear, dlinear),
    Activation.SIGMOID: (sigmoid, dsigmoid),
    Activation.TANH: (tanh, dtanh),
    Activation.RELU: (ReLU, dReLU),
    Activation.SOFTMAX: (softmax, dsoftmax)
}