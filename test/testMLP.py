from MLP import MLP
from MLP.loss import MSE
from MLP.activation import Activation
from MLP.initialization import WeightInitialization
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

bc: dict = load_breast_cancer()
data_bc: np.ndarray = bc['data']
target_bc: np.ndarray = bc['target']

def test_MLP_classification():
    data, test, y_train, y_test = train_test_split(data_bc, target_bc, test_size=0.2, random_state=34)

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    data = (data - data_mean)/data_std
    test = (test - data_mean)/data_std


    # one hot encoding
    n_values = np.max(y_train) + 1
    target = np.eye(n_values)[y_train]

    mlp = MLP(data,
              target,
              hidden_layers=[30, 30],
              weight_initialization=WeightInitialization.HE,
              random_seed=2137,
              activation_functions=[Activation.SIGMOID, Activation.SIGMOID, Activation.SOFTMAX],
              problem_type=MLP.ProblemType.CLASSIFICATION,
              loss_function=MSE)


    mlp.fit(reg_L1=0,
            reg_L2=0,
            batch_size=30,
            epochs=100,
            learning_rate=3e-3,
            momentum=0)
    y_test_predict = mlp.predict(test)
    print()
    print(np.mean(y_test_predict == y_test))