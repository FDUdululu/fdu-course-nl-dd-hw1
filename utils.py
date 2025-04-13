import os
import pickle
import numpy as np
from typing import Dict, Tuple, Callable


class DataLoader:

    @staticmethod
    def load_cifar10_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:

        # TODO

        with open(file_path, 'rb') as file:
            dict = pickle.load(file, encoding='latin1')
            data = dict['data']
            labels = dict['labels']

        labels = np.array(labels)

        return data, labels

    @staticmethod
    def load_cifar10(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # train data
        X_train = []
        y_train = []

        for i in range(1, 6):
            batch_path = os.path.join(data_dir, f'data_batch_{i}')
            X_batch, y_batch = DataLoader.load_cifar10_batch(batch_path)
            X_train.append(X_batch)
            y_train.append(y_batch)

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        # test data
        test_path = os.path.join(data_dir, 'test_batch')
        X_test, y_test = DataLoader.load_cifar10_batch(test_path)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def preprocess_data(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255.0
        X_test /= 255.0

        mean = np.mean(X_train, axis=0)
        X_train -= mean
        X_test -= mean

        return X_train, X_test, mean

    @staticmethod
    def split_validation(X_train: np.ndarray, y_train: np.ndarray,
                         validate_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        num_train = X_train.shape[0]
        num_validate = int(num_train * validate_ratio)
        indices = np.random.permutation(num_train)
        validate_indices = indices[:num_validate]
        train_indices = indices[num_validate:]

        X_validate = X_train[validate_indices]
        y_validate = y_train[validate_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        return X_train, y_train, X_validate, y_validate


class Activation:
    """
    The set of activation functions and their derivatives:

        - relu
        - tanh
        - sigmoid
        - softmax

    These activation functions will be used in the hidden layer.
    """

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # prevent overflow

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        sigmoid = Activation.sigmoid(x)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=1, keepdims=True)  # prevent overflow
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # TODO
    @staticmethod
    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        return 1

    @staticmethod
    def get_activation(function_name: str) -> Tuple[Callable, Callable]:
        name = function_name.lower()
        if name == 'sigmoid':
            return Activation.sigmoid, Activation.sigmoid_derivative
        elif name == 'relu':
            return Activation.relu, Activation.relu_derivative
        elif name == 'tanh':
            return Activation.tanh, Activation.tanh_derivative
        elif name == 'softmax':
            return Activation.softmax, Activation.softmax_derivative
        else:
            raise ValueError(f"Undefined activation function: {function_name}")


class NeuralNetwork:
    """
    Three-layer Neural Network Image Classifier

        Input Layer
            |  linear transform + custom activation function
        Hidden Layer (only one hidden layer)
            |  linear transform + softmax activation function
        Output Layer

    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 hidden_activation: str = 'relu', weight_init_std: float = 0.01):
        """
        Initialize the neural network

        Args:
            input_size: The dimension of the input features
            hidden_size: The number of neurons in the hidden layer
            output_size: The number of output categories
            hidden_activation: The type of activation function for the hidden layer
            weight_init_std: The standard deviation for weight initialization
        """
        # parameters
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # activation function
        self.hidden_activation, self.hidden_activation_derivative = Activation.get_activation(
            hidden_activation)
        # training process cache
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        The output is the probability of each class.
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # Layer 1->2：Linear Transform + Custom Activation Function
        z1 = np.dot(x, W1) + b1
        h = self.hidden_activation(z1)

        # Layer 2->3：Linear Transform + Softmax Activation Function
        z2 = np.dot(h, W2) + b2
        y = Activation.softmax(z2)

        # training mode
        if training:
            self.cache['x'] = x
            self.cache['z1'] = z1
            self.cache['h'] = h
            self.cache['z2'] = z2
            self.cache['y'] = y

        return y

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, regular_lambda: float = 0.0) -> Dict[str, np.ndarray]:

        batch_size = y_true.shape[0]
        x = self.cache['x']
        z1 = self.cache['z1']
        h = self.cache['h']

        # Compute the gradient of the output layer
        # the gradient of (softmax + cross-entropy) is  y_pred - y_true
        dz2 = y_pred - y_true

        # Update the gradient of the weights for the second layer
        dW2 = np.dot(h.T, dz2) / batch_size + \
            regular_lambda * self.params['W2']  # don't forget the gradient of L2 regularization term
        db2 = np.sum(dz2, axis=0) / batch_size

        # Backpropagate to the hidden layer
        dh = np.dot(dz2, self.params['W2'].T)
        dz1 = dh * self.hidden_activation_derivative(z1)

        # Update the gradient of the weights for the first layer
        dW1 = np.dot(x.T, dz1) / batch_size + \
            regular_lambda * self.params['W1']  # don't forget the gradient of L2 regularization term
        db1 = np.sum(dz1, axis=0) / batch_size

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }
        return grads

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = self.forward(x, training=False)
        return np.argmax(y, axis=1)

    def save_model(self, file_path: str) -> None:
        np.savez(file_path, **self.params)

    def load_model(self, file_path: str) -> None:
        data = np.load(file_path)
        for key in self.params:
            self.params[key] = data[key]
