import numpy as np

def MSE(x, y):
    return ((x - y).T @ (x - y)) / len(x)

def CROSSENTROPY(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
