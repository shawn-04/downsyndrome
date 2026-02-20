"""
train.py — Training orchestrator.

Can be run directly:
    uv run python -m src.train

Or imported and called programmatically from notebooks / Streamlit.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from src.config import (
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, RANDOM_SEED,
    BEST_MODEL_PATH, FINAL_MODEL_PATH,
)
from src.data import (
    create_training_generator,
    create_validation_generator,
    create_test_generator,
)
from src.model import build_custom_cnn, build_transfer_learning_cnn, compile_model
from src.evaluate import (
    predict_on_generator,
    get_classification_report,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------

def create_training_callbacks() -> list[callbacks.Callback]:
    """
    Build the standard callback list:
        1. EarlyStopping   — monitors val_auc, patience 8
        2. ReduceLROnPlateau — halves LR after 4 stale epochs
        3. ModelCheckpoint — saves best model by val_auc
    """
    log_dir = str(BEST_MODEL_PATH.parent / "tensorboard_logs")

    return [
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,        # log weight histograms every epoch
            write_graph=True,
            update_freq="epoch",
        ),
    ]


# ---------------------------------------------------------------------------
# Class-weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(generator) -> dict[int, float]:
    """
    Compute balanced class weights from a Keras DataFrameIterator.

    The generator's `.classes` attribute is a 1-D numpy array of integer
    labels (0 or 1) for every sample.

    Returns
    -------
    dict  e.g. {0: 1.2, 1: 0.85}
    """
    total = generator.samples
    unique_classes, class_counts = np.unique(generator.classes, return_counts=True)

    print(f"   Class indices : {generator.class_indices}")
    print(f"   Unique labels : {unique_classes}")
    print(f"   Counts        : {class_counts}")

    weight_dict = {}
    for cls, count in zip(unique_classes, class_counts):
        weight_dict[int(cls)] = total / (len(unique_classes) * count)

    return weight_dict


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def run_training(
    use_transfer_learning: bool = False,
    train_dir: str | None = None,
    valid_dir: str | None = None,
    test_dir: str | None = None,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
):
    """
    End-to-end training pipeline.

    Parameters
    ----------
    use_transfer_learning : bool
        If True, use MobileNetV2 backbone; otherwise the custom 4-block CNN.
    train_dir / valid_dir / test_dir : str | None
        Override default data directories from config.
    epochs : int
        Maximum training epochs (early stopping may cut this short).
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Initial Adam learning rate.

    Returns
    -------
    dict with keys:
        "model", "history", "test_metrics",
        "predicted_probabilities", "true_labels"
    """
    set_global_seed()

    # ---- Data ----------------------------------------------------------
    print("\n📂  Loading data …")
    training_generator   = create_training_generator(train_dir, batch_size)
    validation_generator = create_validation_generator(valid_dir, batch_size)
    test_generator       = create_test_generator(test_dir, batch_size)

    print(f"   Training   : {training_generator.samples} images")
    print(f"   Validation : {validation_generator.samples} images")
    print(f"   Test       : {test_generator.samples} images")
    print(f"   Class map  : {training_generator.class_indices}")

    # ---- Model ---------------------------------------------------------
    print("\n🏗️  Building model …")
    if use_transfer_learning:
        model = build_transfer_learning_cnn()
        print("   Architecture: MobileNetV2 (transfer learning)")
    else:
        model = build_custom_cnn()
        print("   Architecture: Custom 4-block CNN")

    compile_model(model, learning_rate)
    model.summary()

    # ---- Class weights -------------------------------------------------
    class_weight_dict = compute_class_weights(training_generator)
    print(f"\n⚖️  Class weights: {class_weight_dict}")

    # ---- Train ---------------------------------------------------------
    print("\n🚀  Training …")
    training_steps   = max(1, training_generator.samples // batch_size)
    validation_steps = max(1, validation_generator.samples // batch_size)

    history = model.fit(
        training_generator,
        steps_per_epoch=training_steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=create_training_callbacks(),
        verbose=1,
    )

    # ---- Evaluate ------------------------------------------------------
    print("\n📊  Evaluating on test set …")
    test_metrics = model.evaluate(test_generator, verbose=1, return_dict=True)
    predicted_probs, true_labels = predict_on_generator(model, test_generator)

    # ---- Save ----------------------------------------------------------
    model.save(str(FINAL_MODEL_PATH))
    print(f"\n💾  Model saved → {FINAL_MODEL_PATH}")

    return {
        "model": model,
        "history": history,
        "test_metrics": test_metrics,
        "predicted_probabilities": predicted_probs,
        "true_labels": true_labels,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Down Syndrome CNN")
    parser.add_argument("--transfer-learning", action="store_true",
                        help="Use MobileNetV2 instead of the custom CNN")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--valid-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    args = parser.parse_args()

    results = run_training(
        use_transfer_learning=args.transfer_learning,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        test_dir=args.test_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Print summary
    print("\n" + "=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    for metric_name, metric_value in results["test_metrics"].items():
        print(f"  {metric_name:<12}: {metric_value:.4f}")
    print("=" * 55)

    report = get_classification_report(
        results["true_labels"],
        results["predicted_probabilities"],
    )
    print(f"\n{report}")