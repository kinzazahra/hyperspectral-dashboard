import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

def show_band(image, band):
    plt.figure()
    plt.imshow(image[:, :, band], cmap='gray')
    plt.title(f"Band {band}")
    plt.colorbar()
    st.pyplot(plt)

def show_ground_truth(gt):
    plt.figure()
    plt.imshow(gt, cmap='jet')
    plt.title("Ground Truth")
    st.pyplot(plt)

def show_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))