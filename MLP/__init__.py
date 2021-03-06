from enum import Enum
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from MLP.activation import Activation, activation_functions_gradients
from MLP.initialization import WeightInitialization, weight_initialization_methods
from MLP.helpers import chunks, regularization_matrix
from tqdm import tqdm
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class MLP:

    class ProblemType(Enum):
        REGRESSION = 0
        CLASSIFICATION = 1

    def __init__(self,
                 data: np.ndarray,
                 target: np.ndarray,
                 hidden_layers: List[int],
                 weight_initialization: WeightInitialization,
                 random_seed: int,
                 activation_functions: List[Activation],
                 problem_type: ProblemType,
                 loss_function):

        self.data = data
        self.target = target
        self.weight_initialization = weight_initialization
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.activation_functions = activation_functions
        self.vectorized_activation_functions = [activation_functions_gradients[af][0] for af in self.activation_functions]
        self.vectorized_gradient_functions = [activation_functions_gradients[af][1] for af in self.activation_functions]
        self.problem_type = problem_type
        self.layer_sizes = [self.data.shape[1]] + hidden_layers + [target.shape[1] if self.problem_type is MLP.ProblemType.CLASSIFICATION else 1]
        self.loss_function = loss_function
        self.loss_values = []
        self.test_loss_values = []
        self.accuracy_values = []

        weights_init = weight_initialization_methods[self.weight_initialization]

        self.weights = weights_init(self.layer_sizes, self.random_seed)

    def fit(self,
            reg_L1: float,
            reg_L2: float,
            batch_size: int,
            epochs: int,
            learning_rate: float,
            momentum: float,
            test_data: np.ndarray = None,
            test_target: np.ndarray = None) -> None:

        momentum_values = [np.zeros((1 + self.layer_sizes[i], self.layer_sizes[i+1])) for i in range(len(self.layer_sizes)-1)]
        reg_matrices = regularization_matrix(self.layer_sizes)

        for i in tqdm(range(epochs)):
            n = self.data.shape[0]
            for subset in chunks(np.random.choice(np.arange(n), size=n, replace=False), batch_size):
                weight_updates = self._backpropagate(self.data[subset], self.target[subset])
                momentum_values = [weight_update + momentum*layer_momentum for weight_update, layer_momentum in zip(weight_updates, momentum_values)]
                self.weights = [
                    weight + learning_rate * weight_update \
                    - reg_L2 * weight * reg_matrix \
                    - reg_L1 * np.sign(weight) * np.min([np.abs(weight), reg_matrix], axis=0)
                    for weight, weight_update, reg_matrix in zip(self.weights, weight_updates, reg_matrices)]
            y_predict, _ = self._feedforward(self.data)
            self.loss_values.append(self.loss_function(self.target, y_predict))
            if test_data is not None and test_target is not None:
                test_y_predict, _ = self._feedforward(test_data)
                self.test_loss_values.append(self.loss_function(test_target, test_y_predict))
            self.accuracy_values.append(np.mean(self.target == y_predict))



    def predict(self, data) -> np.ndarray:
        if self.problem_type is MLP.ProblemType.CLASSIFICATION:
            return np.argmax(self._feedforward(data)[0], axis=1)
        return self._feedforward(data)[0]

    def predict_proba(self, data) -> np.ndarray:
        return self._feedforward(data)[0]

    # region PLOTS

    def print_loss_by_epoch(self, title=""):
        loss_by_epoch = np.mean(np.abs(self.loss_values), axis=(1, 2)) if len(np.shape(self.loss_values)) == 3 \
            else self.loss_values
        plt.figure(figsize=(10,6))
        plt.plot(loss_by_epoch, label="train loss")
        if len(self.test_loss_values) > 0:
            test_loss_by_epoch = np.mean(np.abs(self.test_loss_values), axis=(1, 2)) if len(np.shape(self.test_loss_values)) == 3 \
                else self.test_loss_values
            plt.plot(test_loss_by_epoch, label="test loss")

        plt.title(title + " Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

    def pca(self,
            predicted_values: np.ndarray = None,
            expected_values: np.ndarray = None,
            test_data: np.ndarray = None):
        # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

        # transforming data to 2D
        d = self.data if predicted_values is None else test_data
        data = StandardScaler().fit_transform(d)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1', 'principal component 2'])

        if predicted_values is not None:
            df_data = expected_values == predicted_values
        elif self.problem_type is MLP.ProblemType.CLASSIFICATION:
            df_data = np.argmax(self.target, axis=1)
        else:
            df_data = self.target
        df = pd.DataFrame(df_data, columns=["target"])
        final_df = pd.concat([principal_df, df], axis=1)

        # plots
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('2 component PCA')

        if self.problem_type is MLP.ProblemType.CLASSIFICATION:
            targets = np.unique(final_df['target'])
            for target in targets:
                indices_to_keep = final_df['target'] == target
                plt.scatter(final_df.loc[indices_to_keep, 'principal component 1'],
                            final_df.loc[indices_to_keep, 'principal component 2'],
                            s=50)
            plt.legend(targets)
        else:
            cmap = sns.cubehelix_palette(as_cmap=True)

            if predicted_values is not None:
                final_target = pd.Series(list(np.abs(predicted_values - expected_values)))
            else:
                final_target = final_df['target']

            plt.scatter(final_df['principal component 1'],
                        final_df['principal component 2'],
                        c=final_target, s=50, cmap=cmap)

    def scatter_classification(self,
                               expected_values: np.ndarray,
                               test_data: np.ndarray,
                               title=None):
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        mesh_step_size = .01  # step size in the mesh
        plot_symbol_size = 50

        x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
        y_min, y_max = test_data[:, 1].min() - 1, test_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                             np.arange(y_min, y_max, mesh_step_size))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(10,6))
        plt.title(title, fontsize=24)
        plt.pcolormesh(xx, yy, Z,
                       cmap=cmap_light)
        # Plot training points
        plt.scatter(test_data[:, 0],
                    test_data[:, 1],
                    s=plot_symbol_size,
                    c=expected_values,
                    cmap=cmap_bold,
                    edgecolor='black')

    def scatter_regression(self,
                           expected_values: np.ndarray,
                           test_data: np.ndarray,
                           title=None):
        plt.figure(figsize=(10,6))
        plt.scatter(test_data, expected_values, alpha=0.4)

        X = np.linspace(test_data.min(), test_data.max(), 100)
        Y = self.predict(X)
        plt.title(title, fontsize=24)
        plt.plot(X, Y)

    # endregion

    def _feedforward(self, input_vector):
        layer_outputs = []
        output_vector = input_vector
        for activation_function, weight_matrix in zip(self.vectorized_activation_functions, self.weights):
            layer_vector = np.c_[np.ones(len(output_vector)), output_vector]
            layer_outputs.append(layer_vector)
            output_vector = activation_function(layer_vector @ weight_matrix)
        layer_outputs.append(output_vector)
        return output_vector, layer_outputs

    def _backpropagate(self, input_vector, y_true):
        y_predicted, layer_outputs = self._feedforward(input_vector)

        if self.activation_functions[-1] is Activation.SOFTMAX:
            delta = self.vectorized_gradient_functions[-1](y_predicted-y_true, layer_outputs[-2] @ self.weights[-1])
            weight_updates = [None for _ in self.weights]
            weight_updates[-1] = - layer_outputs[-2].T @ delta
        else:
            delta = (y_predicted - y_true) * self.vectorized_gradient_functions[-1](layer_outputs[-2] @ self.weights[-1])
            weight_updates = [None for _ in self.weights]
            weight_updates[-1] = - layer_outputs[-2].T @ delta

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T)[:, 1:] * self.vectorized_gradient_functions[i](
                layer_outputs[i] @ self.weights[i])
            weight_updates[i] = - layer_outputs[i].T @ delta

        return weight_updates

    def visualize_architecture(self):
        G = nx.DiGraph()
        edges = []
        baseline = 0
        for layer in self.weights:
            n, m = layer.shape
            for i in range(n):
                for j in range(m):
                    edges.append([baseline+i, baseline+n+j+1, layer[i,j]])
            baseline += n
        G.add_weighted_edges_from(edges)
        pos = graphviz_layout(G, prog='dot', args="-Grankdir=LR")
        nx.draw(G, with_labels=True, pos=pos, font_weight='bold')
        plt.show()
