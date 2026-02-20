"""
gradcam.py — Grad-CAM heatmap generation and overlay utilities.

Highlights which facial regions the CNN attends to when making predictions.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH


def generate_heatmap(
    model: models.Model,
    input_image: np.ndarray,
    target_layer_name: str = "conv_block4",
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model : keras.Model
        Trained CNN with at least one Conv2D layer.
    input_image : np.ndarray
        Preprocessed image of shape (1, H, W, 3) with values in [0, 1].
    target_layer_name : str
        Name of the convolutional layer to inspect.

    Returns
    -------
    np.ndarray
        Heatmap of shape (layer_H, layer_W) normalised to [0, 1].
    """
    grad_model = models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(target_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(input_image)
        loss = prediction[:, 0]

    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Resize a Grad-CAM heatmap and blend it onto the original image.

    Parameters
    ----------
    original_image : np.ndarray
        Original image with pixel values in [0, 1], shape (H, W, 3).
    heatmap : np.ndarray
        Raw heatmap from ``generate_heatmap()``.
    alpha : float
        Opacity of the heatmap overlay.

    Returns
    -------
    np.ndarray
        Blended image of shape (H, W, 3) with values in [0, 1].
    """
    height, width = original_image.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = heatmap_colored.astype(np.float32) / 255.0

    # OpenCV loads as BGR — convert to RGB for display
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = heatmap_colored * alpha + original_image * (1 - alpha)
    return np.clip(blended, 0, 1)


def compute_gradcam_for_image(
    model: models.Model,
    image_array: np.ndarray,
    target_layer_name: str = "conv_block4",
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    End-to-end Grad-CAM: returns the overlay, raw heatmap, and prediction.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    image_array : np.ndarray
        Single image, shape (H, W, 3), pixel values in [0, 1].
    target_layer_name : str
        Convolutional layer to visualise.
    alpha : float
        Heatmap overlay opacity.

    Returns
    -------
    overlay : np.ndarray
        Blended Grad-CAM image (H, W, 3).
    heatmap : np.ndarray
        Raw heatmap (layer_H, layer_W).
    prediction_probability : float
        Model's predicted probability for the positive class.
    """
    input_batch = np.expand_dims(image_array, axis=0)

    heatmap = generate_heatmap(model, input_batch, target_layer_name)
    overlay = overlay_heatmap(image_array, heatmap, alpha)
    prediction_probability = float(model.predict(input_batch, verbose=0)[0][0])

    return overlay, heatmap, prediction_probability