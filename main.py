
import os
import argparse

from trainer import Trainer
from param_search import param_search
from utils import NeuralNetwork, DataLoader
from tester import evaluate_model, visualize_results, visualize_misclassifications

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def main():
    parser = argparse.ArgumentParser(
        description='Three-layer neural network CIFAR-10 classifier')
    parser.add_argument('--data_dir', type=str, default='./data/cifar-10-batches-py',
                        help='Path to the CIFAR-10 dataset directory')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'param_search', 'test'],
                        help='Mode of operation: train, param_search, or test')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.npz',
                        help='Path to the model checkpoint (for test mode)')
    parser.add_argument('--figures', type=str, default='./figs',
                        help='Path to save figures')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Size of the hidden layer')
    parser.add_argument('--regular_lambda', type=float, default=0.0001,
                        help='Weight of L2 regularization')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for the hidden layer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    args = parser.parse_args()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    print("Loading CIFAR-10 dataset...")
    X_train, y_train, X_test, y_test = DataLoader.load_cifar10(args.data_dir)

    print("Preprocessing data...")
    X_train, X_test, X_train_mean = DataLoader.preprocess_data(X_train, X_test)

    X_train, y_train, X_validate, y_validate = DataLoader.split_validation(
        X_train, y_train, validate_ratio=0.1)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_validate.shape}")
    print(f"Test data shape: {X_test.shape}")

    input_size = X_train.shape[1]  # 3072 for CIFAR-10 (32x32x3)
    output_size = 10  # CIFAR-10 has 10 classes.

    if args.mode == 'param_search':
        print("Performing hyperparameter search...")
        best_params = param_search(
            X_train, y_train, X_validate, y_validate, input_size, output_size)

        # update best_params
        args.learning_rate = best_params['learning_rate']
        args.hidden_size = best_params['hidden_size']
        args.regular_lambda = best_params['regular_lambda']
        args.activation = best_params['activation']

        print(f"Using best parameters for training: learning_rate={args.learning_rate}, "
              f"hidden_size={args.hidden_size}, regular_lambda={args.regular_lambda}, "
              f"activation={args.activation}")

    if args.mode in ['train', 'param_search']:
        print("Creating neural network...")
        network = NeuralNetwork(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            hidden_activation=args.activation
        )

        print("Creating trainer...")
        trainer = Trainer(
            network=network,
            learning_rate=args.learning_rate,
            regular_lambda=args.regular_lambda,
            batch_size=args.batch_size,
            checkpoint_dir='./checkpoints'
        )

        print(f"Training for {args.epochs} epochs...")
        history = trainer.train(
            X_train, y_train,
            X_validate, y_validate,
            epochs=args.epochs,
            verbose=True
        )

        trainer.plot_history(history)
        trainer.visualize_weights()

        print("Evaluating on test set...")
        results = evaluate_model(network, X_test, y_test)

        visualize_results(y_test, results['predictions'], CIFAR10_CLASSES)
        visualize_misclassifications(
            X_test, y_test, results['predictions'], X_train_mean, CIFAR10_CLASSES)

    elif args.mode == 'test':
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        network = NeuralNetwork(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            hidden_activation=args.activation
        )
        network.load_model(args.checkpoint)

        print("Evaluating on test set...")
        results = evaluate_model(network, X_test, y_test)

        visualize_results(y_test, results['predictions'], CIFAR10_CLASSES)
        visualize_misclassifications(
            X_test, y_test, results['predictions'], X_train_mean, CIFAR10_CLASSES)


if __name__ == '__main__':
    main()
