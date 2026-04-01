# app/app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Ensure the app can find the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_trained_model
from src.data_loader import preprocess_image_for_inference
from src.metrics import make_gradcam_heatmap, generate_gradcam_overlay

# --- App Configuration ---
st.set_page_config(page_title="Brain Tumor Diagnostic AI", layout="wide")
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] # Ensure these are lowercase if your training folders were lowercase!

# Use os.path to build an absolute, fail-proof path to your model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'advanced_densenet.keras')

# --- Load Model (Cached so it doesn't reload on every click) ---
@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)

model = get_model()

# --- UI Layout ---
st.title("🧠 Brain Tumor Diagnostic Assistant with Explainable AI")
st.markdown("""
Upload a Brain MRI scan. The AI will classify the tumor type and generate a **Grad-CAM heatmap** to highlight the specific region of the brain that influenced its decision.
""")

uploaded_file = st.file_uploader("Choose an MRI image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # Read as BGR
    
    st.markdown("### Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    # Show Original
    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Upload", use_column_width=True)
        
    with st.spinner("Analyzing scan and generating heatmap..."):
        # 2. Preprocess (Crop and Resize)
        img_array, processed_img = preprocess_image_for_inference(img)
        
        # 3. Generate Prediction and Heatmap
        heatmap, predictions = make_gradcam_heatmap(img_array, model)
        
        # Extract results
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100
        
        # 4. Generate Overlay
        overlay_img = generate_gradcam_overlay(processed_img, heatmap)
        
    # Show Processed (Cropped)
    with col2:
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Preprocessed (Cropped)", use_column_width=True)
        
    # Show Grad-CAM
    with col3:
        st.image(overlay_img, caption="Grad-CAM Heatmap", use_column_width=True)
        
    # Final Verdict Box
    st.success(f"**Diagnosis:** {predicted_class_name} | **Confidence:** {confidence:.2f}%")