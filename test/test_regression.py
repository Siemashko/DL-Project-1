from MLP import MLP
from MLP.loss import MSE
from MLP.activation import Activation
from MLP.initialization import WeightInitialization
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def testdata_MPL_regression(data: np.ndarray,
                            y_train: np.ndarray,
                            test: np.ndarray,
                            y_test: np.ndarray,
                            hidden_layers: List[int],
                            weight_initialization: WeightInitialization,
                            random_seed: int,
                            activation_functions: List[Activation],
                            loss_function,
                            reg_L1: float,
                            reg_L2: float,
                            batch_size: int,
                            epochs: int,
                            learning_rate: float,
                            momentum: float,
                            show_plots: bool) -> Tuple[np.ndarray, np.ndarray]:

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    data = (data - data_mean) / data_std
    test = (test - data_mean) / data_std

    mlp = MLP(data,
              y_train.reshape(-1, 1),
              hidden_layers=hidden_layers,
              weight_initialization=weight_initialization,
              random_seed=random_seed,
              activation_functions=activation_functions,
              problem_type=MLP.ProblemType.REGRESSION,
              loss_function=loss_function)
    y_test_predict = mlp.predict(test)
    print(f"\n\nInitial MSE: {MSE(y_test_predict, y_test.reshape(-1, 1))}")

    mlp.fit(reg_L1=reg_L1,
            reg_L2=reg_L2,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            test_data=test,
            test_target=y_test.reshape(-1, 1))

    y_test_predict = mlp.predict(test)
    print(f"\n\nPost training MSE: {MSE(y_test_predict, y_test.reshape(-1, 1))}\n")

    if show_plots:
        mlp.scatter_regression(y_test, test)
        mlp.print_loss_by_epoch()
        plt.show()

    return mlp.loss_values, mlp.accuracy_values


# data preparation
df: pd.DataFrame = pd.read_csv("../../Project 1 datasets/regression/data.cube.train.100.csv")
data: np.ndarray = df[["x"]].to_numpy()
target: np.ndarray = df["y"].to_numpy()
df = pd.read_csv("../../Project 1 datasets/regression/data.cube.test.100.csv")
test_data: np.ndarray = df[["x"]].to_numpy()
test_target: np.ndarray = df["y"].to_numpy()

# sample call
testdata_MPL_regression(data=data,
                        y_train=target,
                        test=test_data,
                        y_test=test_target,
                        hidden_layers=[5, 5],
                        weight_initialization=WeightInitialization.HE,
                        random_seed=2021,
                        activation_functions=[Activation.SIGMOID, Activation.SIGMOID, Activation.LINEAR],
                        loss_function=MSE,
                        reg_L1=0,
                        reg_L2=0,
                        batch_size=2,
                        epochs=100,
                        learning_rate=3e-3,
                        momentum=0,
                        show_plots=True)
