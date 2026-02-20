"""
model.py — CNN architectures for Down Syndrome detection.

Two builders are provided:
    • build_custom_cnn()            → lightweight 4-block CNN from scratch
    • build_transfer_learning_cnn() → MobileNetV2 backbone (ImageNet weights)
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import INPUT_SHAPE


# ---------------------------------------------------------------------------
# Custom CNN (from scratch)
# ---------------------------------------------------------------------------

def build_custom_cnn(input_shape: tuple = INPUT_SHAPE) -> models.Sequential:
    """
    4-block convolutional network for binary classification.

    Architecture per block:
        Conv2D → BatchNorm → ReLU → MaxPool2D → Dropout

    Followed by:
        GlobalAveragePooling → Dense(256) → Dropout → Sigmoid

    Parameters
    ----------
    input_shape : tuple
        Shape of one input image, e.g. (224, 224, 3).

    Returns
    -------
    keras.Sequential
        Un-compiled model ready for .compile().
    """
    filter_counts = [32, 64, 128, 256]
    model = models.Sequential(name="DownSyndrome_CustomCNN")

    for block_index, num_filters in enumerate(filter_counts, start=1):
        conv_kwargs = {"padding": "same", "name": f"conv_block{block_index}"}
        if block_index == 1:
            conv_kwargs["input_shape"] = input_shape

        model.add(layers.Conv2D(num_filters, (3, 3), **conv_kwargs))
        model.add(layers.BatchNormalization(name=f"bn_block{block_index}"))
        model.add(layers.Activation("relu", name=f"relu_block{block_index}"))
        model.add(layers.MaxPooling2D((2, 2), name=f"pool_block{block_index}"))
        model.add(layers.Dropout(0.25, name=f"drop_block{block_index}"))

    # Classification head
    model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))
    model.add(layers.Dense(256, activation="relu", name="dense_hidden"))
    model.add(layers.Dropout(0.5, name="drop_dense"))
    model.add(layers.Dense(1, activation="sigmoid", name="output_sigmoid"))

    return model


# ---------------------------------------------------------------------------
# Transfer-learning CNN (MobileNetV2)
# ---------------------------------------------------------------------------

def build_transfer_learning_cnn(
    input_shape: tuple = INPUT_SHAPE,
    freeze_base: bool = True,
) -> models.Sequential:
    """
    Binary classifier built on top of MobileNetV2 pre-trained on ImageNet.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input image.
    freeze_base : bool
        If True, the MobileNetV2 backbone weights are frozen so only the
        classification head is trained.  Set to False for full fine-tuning.

    Returns
    -------
    keras.Sequential
        Un-compiled model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    transfer_model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(name="gap"),
            layers.Dense(128, activation="relu", name="dense_tl"),
            layers.Dropout(0.5, name="drop_tl"),
            layers.Dense(1, activation="sigmoid", name="output_tl"),
        ],
        name="DownSyndrome_MobileNetV2",
    )

    return transfer_model


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------

def compile_model(
    model: models.Model,
    learning_rate: float = 1e-4,
) -> models.Model:
    """
    Compile the model with Adam, binary cross-entropy, and common metrics.

    Parameters
    ----------
    model : keras.Model
        Any un-compiled Keras model with a sigmoid output.
    learning_rate : float
        Initial Adam learning rate.

    Returns
    -------
    keras.Model
        The same model, now compiled in-place.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model