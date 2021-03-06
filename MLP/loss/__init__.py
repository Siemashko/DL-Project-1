import numpy as np

MSE = lambda x, y: ((x - y).T @ (x - y)) / len(x)

CROSSENTROPY = lambda y_true, y_pred: -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
