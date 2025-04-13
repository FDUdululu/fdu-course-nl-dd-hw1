import os
import time
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

from utils import NeuralNetwork


class Trainer:
    def __init__(self, network: NeuralNetwork,
                 learning_rate: float = 0.1,
                 lr_decay: float = 0.95,
                 regular_lambda: float = 0.00001,
                 batch_size: int = 512,
                 checkpoint_dir: str = './checkpoints'):
        """
        Initialize the trainer

        Args:
            network: The neural network model to be trained
            learning_rate: The initial learning rate
            lr_decay: The learning rate decay factor
            regular_lambda: The strength of L2 regularization
            batch_size: The size of the mini-batch
            checkpoint_dir: The directory to save model checkpoints
        """
        self.network = network
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.regular_lambda = regular_lambda
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

    def cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1.0)  # Prevent numerical instability
        batch_size = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size

        # L2 regularization
        l2_loss = 0.0
        for param_name, param in self.network.params.items():
            if param_name.startswith('W'):  # Regularize only the weights
                l2_loss += 0.5 * self.regular_lambda * np.sum(param ** 2)

        return loss + l2_loss

    def one_hot_encode(self, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """
        Convert class labels to one-hot encoding.
        Return y_true, size: (batch_size, num_classes)
        """
        y_true = np.zeros((y.shape[0], num_classes))
        y_true[np.arange(y.shape[0]), y] = 1
        return y_true

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(y_pred == y_true)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_validate: np.ndarray, y_validate: np.ndarray,
              epochs: int = 10,
              verbose: bool = True,
              early_stopping_patience: int = 5) -> Dict:
        """
        Train the neural network

        Args:
            X_train: Training set features
            y_train: Training set labels
            X_validate: Validation set features
            y_validate: Validation set labels
            epochs: Number of training epochs
            verbose: Whether to print training information
            early_stopping_patience: Patience value for early stopping

        Returns:
            Training history
        """
        num_train = X_train.shape[0]
        # The CIFAR-10 dataset consists of 10 classes, with indices starting from 0.
        num_classes = 10

        # training history
        history = {
            'train_loss': [],
            'validate_loss': [],
            'validate_accuracy': [],
            'best_validate_accuracy': 0.0,
            'best_epoch': 0
        }

        # Early stopping counter
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()
            current_lr = self.learning_rate * (self.lr_decay ** epoch)

            # Shuffle the training data
            indices = np.random.permutation(num_train)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Mini-batch training
            train_loss = 0
            num_batches = int(np.ceil(num_train / self.batch_size))

            # train for each batch
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, num_train)

                # Convert class labels to one-hot encoding
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                y_batch_one_hot = self.one_hot_encode(y_batch, num_classes)

                # forward propagation
                y_pred = self.network.forward(X_batch)

                # calculate loss
                batch_loss = self.cross_entropy_loss(y_pred, y_batch_one_hot)
                train_loss += batch_loss

                # backward propagation
                grads = self.network.backward(
                    y_pred, y_batch_one_hot, self.regular_lambda)

                # update params
                for param_name in self.network.params:
                    self.network.params[param_name] -= current_lr * \
                        grads[param_name]

            # train loss
            train_loss /= num_batches

            # validate loss
            y_validate_one_hot = self.one_hot_encode(y_validate, num_classes)
            y_validate_pred_prob = self.network.forward(
                X_validate, training=False)
            validate_loss = self.cross_entropy_loss(
                y_validate_pred_prob, y_validate_one_hot)

            # validate accuracy
            y_validate_pred = np.argmax(y_validate_pred_prob, axis=1)
            validate_accuracy = self.accuracy(y_validate_pred, y_validate)

            # epoch training time
            epoch_time = time.time() - start_time

            if verbose:
                print(f"Epoch: {epoch + 1:02d}/{epochs:02d} - epoch_time: {epoch_time:.2f}s - " +
                      f"train_loss: {train_loss:.4f} - validate_loss: {validate_loss:.4f} - " +
                      f"validate_accuracy: {validate_accuracy:.4f} - learning_rate: {current_lr:.4f}")

            history['train_loss'].append(train_loss)
            history['validate_loss'].append(validate_loss)
            history['validate_accuracy'].append(validate_accuracy)
            # Best Model
            if validate_accuracy > history['best_validate_accuracy']:
                history['best_validate_accuracy'] = validate_accuracy
                history['best_epoch'] = epoch
                self.network.save_model(os.path.join(
                    self.checkpoint_dir, 'best_model.npz'))
                patience_counter = 0
            else:
                patience_counter += 1
            """
            If the model's performance on the validation set does not improve 
                for a number of consecutive epochs exceeding the patience value, 
                    early stopping is triggered.
            """
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def plot_history(self, history: Dict) -> None:
        epochs = range(1, len(history['train_loss']) + 1)

        plt.figure(figsize=(12, 4))

        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='train loss')
        plt.plot(epochs, history['validate_loss'], 'r-', label='validate loss')
        plt.title('Train Loss and Validate Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['validate_accuracy'],
                 'g-', label='validate accuracy')
        plt.axvline(x=history['best_epoch']+1, color='r', linestyle='--',
                    label=f'Best Model (Epoch {history["best_epoch"]+1})')
        plt.title('Validate Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./figs/training_history.png')
        plt.close()

    def visualize_weights(self) -> None:
        """
        Visualize the weights of the first layer of the network
        """
        W1 = self.network.params['W1']
        num_neurons = min(25, W1.shape[1])  # at most 25 neurons

        plt.figure(figsize=(12, 12))
        for i in range(num_neurons):
            plt.subplot(5, 5, i+1)
            # Assuming the input is a 3072-dimensional CIFAR-10 image (32x32x3)
            # reshape to 32x32x3
            weight = W1[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
            # normalize
            weight = (weight - weight.min()) / (weight.max() - weight.min())
            plt.imshow(weight)
            plt.axis('off')
            plt.title(f'Neuron {i + 1}')

        plt.tight_layout()
        plt.savefig('./figs/weight_visualization.png')
        plt.close()
