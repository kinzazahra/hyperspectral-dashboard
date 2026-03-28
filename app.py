import streamlit as st
import numpy as np
import pickle

from utils.data_loader import load_mat_file, extract_data
from utils.preprocessing import flatten_data, split_data, apply_pca
from utils.model import get_model, train_model
from utils.visualization import show_band, show_ground_truth, show_confusion_matrix

st.set_page_config(page_title="HSI Dashboard", layout="wide")

st.title("🛰️ Hyperspectral Image Analysis Dashboard")

# Sidebar
st.sidebar.header("⚙️ Settings")

use_pca = st.sidebar.checkbox("Enable PCA")
n_components = st.sidebar.slider("PCA Components", 5, 50, 20)

model_type = st.sidebar.selectbox("Model", ["knn", "naive_bayes"])
k = st.sidebar.slider("k (for kNN)", 1, 10, 3)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)

# Upload files
image_file = st.file_uploader("Upload HSI Image (.mat)")
gt_file = st.file_uploader("Upload Ground Truth (.mat)")

if image_file and gt_file:
    image_data = load_mat_file(image_file)
    gt_data = load_mat_file(gt_file)

    image = extract_data(image_data)
    gt = extract_data(gt_data)

    st.success("Files loaded successfully!")

    # Show info
    st.write("Image Shape:", image.shape)

    # Visualization
    st.subheader("📊 Exploratory Data Analysis")

    band = st.slider("Select Band", 0, image.shape[2]-1)
    show_band(image, band)

    show_ground_truth(gt)

    # Preprocessing
    X, y = flatten_data(image, gt)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    if use_pca:
        X_train, X_test, pca = apply_pca(X_train, X_test, n_components)
        st.write("PCA Applied")

    # Train
    if st.button("🚀 Train Model"):
        model = get_model(model_type, k)
        model = train_model(model, X_train, y_train)

        y_pred = model.predict(X_test)

        acc = np.mean(y_pred == y_test)

        st.success(f"Accuracy: {acc:.4f}")

        show_confusion_matrix(y_test, y_pred)

        # Save model
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        st.success("Model saved in models/model.pkl")

