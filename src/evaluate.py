"""
evaluate.py — Model evaluation utilities: metrics, plots, and reports.

All plotting functions return (fig, ax) so callers can either display
them interactively or save to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
)
from tensorflow.keras import Model

from src.config import CLASS_NAMES


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_on_generator(model: Model, generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on an entire generator and return probabilities + true labels.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    generator : DirectoryIterator
        Keras data generator (must have shuffle=False).

    Returns
    -------
    predicted_probabilities : np.ndarray   shape (N,)
    true_labels             : np.ndarray   shape (N,)
    """
    generator.reset()
    predicted_probabilities = model.predict(generator, verbose=1).ravel()
    true_labels = generator.classes
    return predicted_probabilities, true_labels


# ---------------------------------------------------------------------------
# Training history curves
# ---------------------------------------------------------------------------

def plot_training_history(
    history,
    metrics: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot training vs. validation curves for selected metrics.

    Parameters
    ----------
    history : keras History object
        Returned by model.fit().
    metrics : list[str] | None
        Metric keys to plot.  Defaults to ["loss", "accuracy", "auc"].

    Returns
    -------
    (fig, axes)
    """
    if metrics is None:
        metrics = ["loss", "accuracy", "auc"]

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        train_values = history.history[metric_name]
        val_values = history.history[f"val_{metric_name}"]
        epoch_range = range(1, len(train_values) + 1)

        ax.plot(epoch_range, train_values, lw=2, label=f"Train {metric_name}")
        ax.plot(epoch_range, val_values, lw=2, ls="--", label=f"Val {metric_name}")
        ax.set_title(f"{metric_name.upper()} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> tuple[plt.Figure, plt.Axes, float]:
    """
    Plot the Receiver Operating Characteristic curve and compute AUC.

    Returns
    -------
    (fig, ax, roc_auc_value)
    """
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    roc_auc_value = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"ROC Curve (AUC = {roc_auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig, ax, roc_auc_value


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Plot a heatmap confusion matrix at the given decision threshold.

    Returns
    -------
    (fig, ax, confusion_matrix_array)
    """
    predicted_classes = (predicted_probabilities >= threshold).astype(int)
    cm = confusion_matrix(true_labels, predicted_classes)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()

    return fig, ax, cm


# ---------------------------------------------------------------------------
# Classification report (text)
# ---------------------------------------------------------------------------

def get_classification_report(
    true_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """Return a sklearn classification report string."""
    predicted_classes = (predicted_probabilities >= threshold).astype(int)
    return classification_report(
        true_labels, predicted_classes, target_names=CLASS_NAMES,
    )