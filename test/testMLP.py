from MLP import MLP
from MLP.loss import MSE, CROSSENTROPY
from MLP.activation import Activation
from MLP.initialization import WeightInitialization
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
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
    test_target = np.eye(n_values)[y_test]

    mlp = MLP(data,
              target,
              hidden_layers=[30, 30],
              weight_initialization=WeightInitialization.HE,
              random_seed=2137,
              activation_functions=[Activation.SIGMOID, Activation.SIGMOID, Activation.SOFTMAX],
              problem_type=MLP.ProblemType.CLASSIFICATION,
              loss_function=CROSSENTROPY)


    mlp.fit(reg_L1=0,
            reg_L2=0,
            batch_size=30,
            epochs=100,
            learning_rate=3e-3,
            momentum=0,
            test_data=test,
            test_target=test_target)
    y_test_predict = mlp.predict(test)
    print()
    print(np.mean(y_test_predict == y_test))

    mlp.pca()
    plt.show()

    mlp.pca(y_test_predict, y_test, test)
    plt.show()

    mlp.print_loss_by_epoch()
    plt.show()

boston: dict = load_boston()
data_boston: np.ndarray = boston['data']
target_boston: np.ndarray = boston['target']

def test_MLP_regression():
    data, test, y_train, y_test = train_test_split(data_boston, target_boston, test_size=0.2, random_state=34)

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    data = (data - data_mean) / data_std
    test = (test - data_mean) / data_std

    mlp = MLP(data,
              y_train.reshape(-1,1),
              hidden_layers=[13, 13],
              weight_initialization=WeightInitialization.HE,
              random_seed=2137,
              activation_functions=[Activation.SIGMOID, Activation.SIGMOID, Activation.LINEAR],
              problem_type=MLP.ProblemType.REGRESSION,
              loss_function=MSE)
    y_test_predict = mlp.predict(test)
    print(f"\n\nInitial MSE: {MSE(y_test_predict, y_test.reshape(-1,1))}")

    mlp.fit(reg_L1=0,
            reg_L2=0,
            batch_size=13,
            epochs=100,
            learning_rate=3e-3,
            momentum=0,
            test_data=test,
            test_target=y_test.reshape(-1,1))

    y_test_predict = mlp.predict(test)
    print(f"\n\nPost training MSE: {MSE(y_test_predict, y_test.reshape(-1,1))}\n")

    mlp.pca()
    plt.show()

    mlp.pca(y_test_predict, y_test.reshape(-1,1), test)
    plt.show()

    mlp.print_loss_by_epoch()
    plt.show()


test_MLP_classification()
test_MLP_regression()