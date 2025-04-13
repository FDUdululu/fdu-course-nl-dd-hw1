
import time
import numpy as np
from typing import Dict

from trainer import Trainer
from utils import NeuralNetwork


def param_search(X_train: np.ndarray, y_train: np.ndarray,
                 X_validate: np.ndarray, y_validate: np.ndarray,
                 input_size: int, output_size: int) -> Dict:
    """
    Hyperparameter Search

    Args:
        X_train: Training set features
        y_train: Training set labels
        X_validate: Validation set features
        y_validate: Validation set labels
        input_size: Input dimension
        output_size: Output dimension

    Returns:
        Best parameter configuration
    """

    # Hyperparameter search space
    learning_rates = [0.001, 0.01, 0.1]
    hidden_sizes = [128, 256, 512]
    regular_lambdas = [0.00001, 0.0001, 0.001, 0.01]
    activations = ['relu', 'tanh']

    # Fixed parameters
    batch_size = 128
    epochs = 5  # Use fewer epochs during the hyperparameter search phase.

    best_validate_accuracy = 0.0
    best_params = None
    results = []

    # To speed up the search, use only a subset of the training data.
    subset_size = min(10000, X_train.shape[0])
    indices = np.random.choice(X_train.shape[0], subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            for regular_lambda in regular_lambdas:
                for activation in activations:
                    print(f"\nTesting parameters: learning_rates={lr}, hidden_size={hidden_size}, " +
                          f"regular_lambda={regular_lambda}, activation={activation}")

                    network = NeuralNetwork(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        hidden_activation=activation
                    )

                    trainer = Trainer(
                        network=network,
                        learning_rate=lr,
                        regular_lambda=regular_lambda,
                        batch_size=batch_size,
                        checkpoint_dir=f'./checkpoints/search_lr{lr}_hidden{hidden_size}_reg{regular_lambda}_act{activation}'
                    )

                    start_time = time.time()
                    history = trainer.train(
                        X_train_subset, y_train_subset,
                        X_validate, y_validate,
                        epochs=epochs,
                        verbose=True,
                        early_stopping_patience=3
                    )
                    training_time = time.time() - start_time
                    validate_accuracy = history['best_validate_accuracy']

                    param_result = {
                        'learning_rate': lr,
                        'hidden_size': hidden_size,
                        'regular_lambda': regular_lambda,
                        'activation': activation,
                        'validate_accuracyuracy': validate_accuracy,
                        'training_time': training_time
                    }
                    results.append(param_result)

                    if validate_accuracy > best_validate_accuracy:
                        best_validate_accuracy = validate_accuracy
                        best_params = param_result

                    print(
                        f"Validation accuracy: {validate_accuracy:.4f}, Training time: {training_time:.2f}s")

    print("")
    print("Best parameters:")
    print(f"Learning rate: {best_params['learning_rate']}")
    print(f"Hidden size: {best_params['hidden_size']}")
    print(f"Regularization lambda: {best_params['regular_lambda']}")
    print(f"Activation: {best_params['activation']}")
    print(f"Validation accuracy: {best_params['validate_accuracyuracy']:.4f}")

    return best_params
