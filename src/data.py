"""
data.py — Data loading, preprocessing, and augmentation pipelines.

Dataset layout
--------------
Each split lives in a flat folder alongside a CSV label file:

    ds/
    ├── train/
    │   ├── _classes.csv
    │   ├── down_1497_jpg.rf.….jpg
    │   └── healty_73_jpg.rf.….jpg
    ├── valid/
    │   ├── _classes.csv
    │   └── …
    └── test/
        ├── _classes.csv
        └── …

The CSV has three columns:
    filename, downSyndrome, healthy

We derive a single binary label:
    label = "Down_Syndrome" if downSyndrome == 1 else "Healthy"

Keras' `flow_from_dataframe` is used to pair each filename with its label.

Exports
-------
    create_training_generator()
    create_validation_generator()
    create_test_generator()
    load_and_prepare_labels()
"""

from pathlib import Path

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR,
    IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED,
    FILENAME_COLUMN, DOWN_SYNDROME_COLUMN,
    CLASS_NAMES,
)


# ---------------------------------------------------------------------------
# CSV → DataFrame helper
# ---------------------------------------------------------------------------

def _find_csv_in_folder(folder: Path) -> Path:
    """
    Locate the label CSV inside a split folder.

    Strategy (in priority order):
        1. _classes.csv  (Roboflow default)
        2. Any single .csv file in the folder
        3. Raise FileNotFoundError
    """
    default_csv = folder / "_classes.csv"
    if default_csv.exists():
        return default_csv

    csv_candidates = list(folder.glob("*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]

    raise FileNotFoundError(
        f"Could not find a label CSV in {folder}. "
        "Expected '_classes.csv' or a single .csv file."
    )


def load_and_prepare_labels(folder: Path) -> pd.DataFrame:
    """
    Read the label CSV and add a human-readable ``label`` column.

    Parameters
    ----------
    folder : Path
        Directory that contains both the CSV and the images.

    Returns
    -------
    pd.DataFrame
        Columns: filename, downSyndrome, healthy, label
        ``label`` is a string — one of config.CLASS_NAMES.
    """
    csv_path = _find_csv_in_folder(folder)
    labels_dataframe = pd.read_csv(csv_path)

    # Strip whitespace from column headers (CSVs can be finicky)
    labels_dataframe.columns = labels_dataframe.columns.str.strip()

    # Derive a single string label for Keras flow_from_dataframe
    labels_dataframe["label"] = labels_dataframe[DOWN_SYNDROME_COLUMN].apply(
        lambda flag: CLASS_NAMES[1] if flag == 1 else CLASS_NAMES[0]
    )

    return labels_dataframe


# ---------------------------------------------------------------------------
# Augmentation parameters (training only)
# ---------------------------------------------------------------------------

def _augmentation_params() -> dict:
    """Return augmentation keyword args applied only to the training split."""
    return dict(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )


# ---------------------------------------------------------------------------
# Generator factories
# ---------------------------------------------------------------------------

def create_training_generator(
    train_dir: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
):
    """
    Augmented training generator built from the CSV + flat image folder.

    Parameters
    ----------
    train_dir : str | Path | None
        Override directory.  Defaults to config.TRAIN_DIR.
    batch_size : int
        Images per batch.

    Returns
    -------
    DataFrameIterator
    """
    folder = Path(train_dir) if train_dir else TRAIN_DIR
    labels_df = load_and_prepare_labels(folder)

    generator = ImageDataGenerator(
        rescale=1.0 / 255.0,
        **_augmentation_params(),
    )

    return generator.flow_from_dataframe(
        dataframe=labels_df,
        directory=str(folder),
        x_col=FILENAME_COLUMN,
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=RANDOM_SEED,
        color_mode="rgb",
    )


def create_validation_generator(
    valid_dir: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
):
    """
    Non-augmented validation generator (pre-split folder).

    Parameters
    ----------
    valid_dir : str | Path | None
        Override directory.  Defaults to config.VALID_DIR.
    batch_size : int
        Images per batch.

    Returns
    -------
    DataFrameIterator
    """
    folder = Path(valid_dir) if valid_dir else VALID_DIR
    labels_df = load_and_prepare_labels(folder)

    generator = ImageDataGenerator(rescale=1.0 / 255.0)

    return generator.flow_from_dataframe(
        dataframe=labels_df,
        directory=str(folder),
        x_col=FILENAME_COLUMN,
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        seed=RANDOM_SEED,
        color_mode="rgb",
    )


def create_test_generator(
    test_dir: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
):
    """
    Non-augmented test generator.

    Parameters
    ----------
    test_dir : str | Path | None
        Override directory.  Defaults to config.TEST_DIR.
    batch_size : int
        Images per batch.

    Returns
    -------
    DataFrameIterator
    """
    folder = Path(test_dir) if test_dir else TEST_DIR
    labels_df = load_and_prepare_labels(folder)

    generator = ImageDataGenerator(rescale=1.0 / 255.0)

    return generator.flow_from_dataframe(
        dataframe=labels_df,
        directory=str(folder),
        x_col=FILENAME_COLUMN,
        y_col="label",
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
        seed=RANDOM_SEED,
        color_mode="rgb",
    )