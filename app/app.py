import sys
import os
import gdown
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# --- CLOUD-SAFE PATH CONFIGURATION ---
# Instead of saving in the project folder, we use the system's temporary directory
MODEL_DIR = "/tmp/saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, 'advanced_densenet.keras')

# Create the temp folder if it doesn't exist (this is allowed in /tmp)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            file_id = '1phkCm78u090s7Otrjy2rOxgp5S9VpyNd'
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            # Download directly to the /tmp path
            gdown.download(url, MODEL_PATH, quiet=False)
    
    from src.model import load_trained_model
    return load_trained_model(MODEL_PATH)

from src.model import load_trained_model
from src.data_loader import preprocess_image_for_inference
from src.metrics import make_gradcam_heatmap, generate_gradcam_overlay

# --- MODEL DOWNLOAD LOGIC ---
MODEL_DIR = os.path.join(root_dir, 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'advanced_densenet.keras')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (127MB)... Please wait."):
            file_id = '1phkCm78u090s7Otrjy2rOxgp5S9VpyNd'
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return load_trained_model(MODEL_PATH)

# --- APP UI ---
st.set_page_config(page_title="Brain Tumor AI", layout="wide")
# --- UI Layout ---
st.title("🧠 Brain Tumor Diagnostic Assistant with Explainable AI")

# FIX: Ensure this is wrapped in triple quotes
st.markdown("""
Upload a Brain MRI scan. The AI will classify the tumor type and generate a **Grad-CAM heatmap** to highlight the specific region of the brain that influenced its decision.
""")

uploaded_file = st.file_uploader("Choose an MRI image (JPG, PNG)", type=["jpg", "jpeg", "png"])

try:
    model = get_model()
    CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col1, col2, col3 = st.columns(3)
    
    with st.spinner("Analyzing..."):
        img_array, processed_img = preprocess_image_for_inference(img)
        heatmap, predictions = make_gradcam_heatmap(img_array, model)
        
        idx = np.argmax(predictions)
        label = CLASS_NAMES[idx].capitalize()
        conf = predictions[idx] * 100
        overlay = generate_gradcam_overlay(processed_img, heatmap)

    with col1: st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    with col2: st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Cropped")
    with col3: st.image(overlay, caption="Grad-CAM Heatmap")

    st.success(f"**Result:** {label} ({conf:.2f}%)")
