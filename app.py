"""
app.py — Streamlit interface for Down Syndrome Detection.

Launch with:
    uv run streamlit run app.py

Pages
-----
1.  Predict   — upload a facial image → prediction + Grad-CAM
2.  Dataset   — explore class distributions & sample images from the CSV
3.  Train     — kick off training from the browser (optional)
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import tensorflow as tf

from src.config import (
    IMAGE_SIZE, CLASS_NAMES,
    FINAL_MODEL_PATH, BEST_MODEL_PATH,
    TRAIN_DIR, VALID_DIR, TEST_DIR,
)
from src.data import load_and_prepare_labels
from src.gradcam import compute_gradcam_for_image


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Down Syndrome Detection",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def load_trained_model(model_path: str) -> tf.keras.Model:
    """Load and cache a trained Keras model from disk."""
    path = Path(model_path)
    if not path.exists():
        return None
    return tf.keras.models.load_model(str(path))


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_uploaded_image(uploaded_file) -> tuple[np.ndarray, Image.Image]:
    """
    Read an uploaded file into a preprocessed numpy array and a PIL image.

    Returns
    -------
    preprocessed_array : np.ndarray
        Shape (IMAGE_H, IMAGE_W, 3), values in [0, 1].
    display_image : PIL.Image
        Resized RGB image for display purposes.
    """
    pil_image = Image.open(uploaded_file).convert("RGB")
    resized_image = pil_image.resize(IMAGE_SIZE)
    image_array = np.array(resized_image, dtype=np.float32) / 255.0
    return image_array, resized_image


# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    page = st.radio(
        "Navigate",
        options=["Predict", "Dataset", "Train"],
        index=0,
    )

    st.divider()

    # Model selector (only relevant for Predict page but always visible)
    available_model_paths = [
        str(p) for p in [BEST_MODEL_PATH, FINAL_MODEL_PATH] if p.exists()
    ]

    if available_model_paths:
        selected_model_path = st.selectbox(
            "Model file", options=available_model_paths, index=0,
        )
    else:
        selected_model_path = None
        st.warning("No trained model found.")

    decision_threshold = st.slider(
        "Decision threshold", 0.0, 1.0, 0.5, 0.05,
        help="Probability above this → classified as Down Syndrome.",
    )

    show_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)

    gradcam_opacity = st.slider(
        "Grad-CAM opacity", 0.1, 0.9, 0.4, 0.05,
        disabled=not show_gradcam,
    )

    st.divider()
    st.caption("Built with TensorFlow & Streamlit")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Predict
# ═══════════════════════════════════════════════════════════════════════════

if page == "Predict":
    st.title("Down Syndrome Detection")
    st.markdown(
        "Upload a facial photograph and the CNN will predict whether "
        "indicators of Down syndrome are present."
    )

    if selected_model_path is None:
        st.error(
            "No trained model found. Go to the **🚀 Train** page or run:\n\n"
            "```bash\nuv run python -m down_syndrome_cnn.train\n```"
        )
        st.stop()

    model = load_trained_model(selected_model_path)
    if model is None:
        st.error("Failed to load the model.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a facial image …",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        preprocessed_image, display_image = preprocess_uploaded_image(uploaded_file)

        # Predict
        input_batch = np.expand_dims(preprocessed_image, axis=0)
        raw_probability = float(model.predict(input_batch, verbose=0)[0][0])

        is_down_syndrome = raw_probability >= decision_threshold
        predicted_label = CLASS_NAMES[1] if is_down_syndrome else CLASS_NAMES[0]
        confidence = raw_probability if is_down_syndrome else 1.0 - raw_probability

        # ---- Layout -------------------------------------------------------
        col_image, col_results = st.columns([1, 1], gap="large")

        with col_image:
            st.subheader("📷 Uploaded Image")
            st.image(display_image, use_container_width=True)

            if show_gradcam:
                try:
                    conv_layer_names = [
                        l.name for l in model.layers
                        if "conv" in l.name.lower()
                    ]
                    if conv_layer_names:
                        target_layer = conv_layer_names[-1]
                        overlay, _, _ = compute_gradcam_for_image(
                            model, preprocessed_image, target_layer, gradcam_opacity,
                        )
                        st.subheader("Grad-CAM")
                        st.image(overlay, clamp=True, use_container_width=True)
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}")

        with col_results:
            st.subheader("Prediction")

            if is_down_syndrome:
                st.error(f"**{predicted_label}**")
            else:
                st.success(f"**{predicted_label}**")

            st.metric("Confidence", f"{confidence:.1%}")
            st.metric("Raw P(Down Syndrome)", f"{raw_probability:.4f}")
            st.metric("Decision threshold", f"{decision_threshold:.2f}")

            st.markdown("##### Probability Distribution")
            col_h, col_ds = st.columns(2)
            with col_h:
                st.metric(CLASS_NAMES[0], f"{1 - raw_probability:.1%}")
                st.progress(1 - raw_probability)
            with col_ds:
                st.metric(CLASS_NAMES[1], f"{raw_probability:.1%}")
                st.progress(raw_probability)

            st.divider()
            st.caption(
                "**Disclaimer** — This tool is a prototype and "
                "is NOT a substitute for professional medical diagnosis."
            )
    else:
        st.info("Upload a facial image to get started.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Dataset
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Dataset":
    st.title("Dataset Explorer")
    st.markdown("Browse the class distributions and sample images from each split.")

    split_tabs = st.tabs(["Train", "Validation", "Test"])
    split_dirs = [TRAIN_DIR, VALID_DIR, TEST_DIR]

    for tab, split_dir in zip(split_tabs, split_dirs):
        with tab:
            if not split_dir.exists():
                st.warning(f"Directory not found: `{split_dir}`")
                continue

            try:
                labels_df = load_and_prepare_labels(split_dir)
            except FileNotFoundError as e:
                st.warning(str(e))
                continue

            # ---- Class distribution -----------------------------------
            class_counts = labels_df["label"].value_counts()
            total_images = len(labels_df)

            col_stats, col_chart = st.columns([1, 2])

            with col_stats:
                st.markdown(f"**Total images:** {total_images}")
                for class_name, count in class_counts.items():
                    pct = count / total_images * 100
                    st.markdown(f"- **{class_name}:** {count} ({pct:.1f}%)")

            with col_chart:
                st.bar_chart(
                    class_counts.rename_axis("Class").reset_index(name="Count"),
                    x="Class",
                    y="Count",
                )

            # ---- Sample images ----------------------------------------
            st.markdown("#### Sample Images")
            num_samples = min(8, total_images)
            sample_rows = labels_df.sample(n=num_samples, random_state=42)

            image_cols = st.columns(4)
            for idx, (_, row) in enumerate(sample_rows.iterrows()):
                image_path = split_dir / row["filename"]
                col = image_cols[idx % 4]
                with col:
                    if image_path.exists():
                        st.image(
                            str(image_path),
                            caption=row["label"],
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Missing: {row['filename']}")

            # ---- Raw CSV preview --------------------------------------
            with st.expander("View raw CSV labels"):
                st.dataframe(labels_df, use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Train
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Train":
    st.title("Train Model")
    st.markdown("Configure and launch training directly from the browser.")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        use_transfer_learning = st.toggle(
            "Use transfer learning (MobileNetV2)", value=False,
        )
        training_epochs = st.number_input(
            "Max epochs", min_value=1, max_value=200, value=50,
        )

    with col_cfg2:
        training_batch_size = st.selectbox(
            "Batch size", options=[8, 16, 32, 64], index=2,
        )
        training_learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
        )

    st.divider()

    if st.button("Start Training", type="primary", use_container_width=True):
        # Import here to avoid loading TF on every page render
        from src.train import run_training

        progress_placeholder = st.empty()
        with st.spinner("Training in progress — check your terminal for live logs …"):
            results = run_training(
                use_transfer_learning=use_transfer_learning,
                epochs=training_epochs,
                batch_size=training_batch_size,
                learning_rate=training_learning_rate,
            )

        st.success("Training complete!")

        # Show test metrics
        st.subheader("Test Set Results")
        metric_cols = st.columns(len(results["test_metrics"]))
        for col, (name, value) in zip(metric_cols, results["test_metrics"].items()):
            col.metric(name.upper(), f"{value:.4f}")

        # Show classification report
        from src.evaluate import get_classification_report
        report = get_classification_report(
            results["true_labels"],
            results["predicted_probabilities"],
        )
        st.subheader("Classification Report")
        st.code(report)

        st.info(f"Model saved to `{FINAL_MODEL_PATH}`")

    else:
        st.info("Press **Start Training** when ready. This may take several minutes.")