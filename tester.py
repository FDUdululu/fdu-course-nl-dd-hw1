
import time
import numpy as np
import seaborn as sns
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from utils import NeuralNetwork


def evaluate_model(network: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    network is a trained model
    Evaluate the model's performance on the test set
    """
    start_time = time.time()

    y_pred = network.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    inference_time = time.time() - start_time

    results = {
        'accuracy': accuracy,
        'inference_time': inference_time,
        'predictions': y_pred
    }

    print(f"Test accuracy: {accuracy:.4f}")
    print(
        f"Inference time: {inference_time:.2f}s for {X_test.shape[0]} samples")

    return results


def visualize_results(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None) -> None:

    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('./figs/confusion_matrix.png')
    plt.close()

    # classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    with open('classification_report.txt', 'w') as file:
        file.write(report)


def visualize_misclassifications(X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, X_train_mean: np.ndarray,
                                 class_names: list = None) -> None:

    if class_names is None:
        class_names = [f'Class {i}' for i in range(10)]

    misclassified_indices = np.where(y_true != y_pred)[0]

    # Display up to 25 misclassified samples at most.
    num_samples = min(25, len(misclassified_indices))
    sample_indices = np.random.choice(
        misclassified_indices, num_samples, replace=False)

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(sample_indices):
        plt.subplot(5, 5, i + 1)
        # Process CIFAR - 10 data format (assuming the data shape is [N, 3072])
        img = X_test[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        # Restore the effects of standardization and centering for display
        # Restore the operation of subtracting the mean
        img += X_train_mean.reshape(3, 32, 32).transpose(1, 2, 0)
        img *= 255.0  # Restore the operation of dividing by 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(
            f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./figs/misclassifications.png')
    plt.close()
